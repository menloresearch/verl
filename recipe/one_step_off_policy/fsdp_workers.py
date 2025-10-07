# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from verl.utils.fsdp_utils import (
    collect_lora_params,
    layered_summon_lora_params,
    replace_lora_wrapper,
    fsdp_version,
)
import dataclasses
from verl.utils.vllm.utils import TensorLoRARequest
import time
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    collect_lora_params,
    layered_summon_lora_params,
    replace_lora_wrapper,
    fsdp_version,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker as ARRWorker
from verl.workers.fsdp_workers import CriticWorker
from verl.workers.rollout import get_rollout_class

from .distributed_util import stateless_init_process_group
from peft import TaskType, get_peft_model
from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
import json
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["ActorRolloutRefWorker", "AsyncActorRolloutRefWorker", "CriticWorker", "RolloutWorker"]


def _convert_sets_to_lists(obj):
    """Recursively convert sets (and nested sets) into lists so JSON can serialize."""
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, tuple):
        return [_convert_sets_to_lists(v) for v in obj]
    if isinstance(obj, list):
        return [_convert_sets_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _convert_sets_to_lists(v) for k, v in obj.items()}
    return obj


class ActorRolloutRefWorker(ARRWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_sync_done = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size):
        rank = torch.distributed.get_rank() + rank_offset
        self._weight_sync_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            get_torch_device().current_device(),
        )

    def _get_actor_params(self):
        assert self._is_actor
        #Check if model has LoRA
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        peft_config = None

        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=False,  # Always False for one-step off-policy
                base_sync_done=self.base_sync_done,
            )
            # On first sync, transform keys to match vLLM's expected format
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params, peft_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        import psutil
        import time as time_module
        
        # Measure network I/O before sync
        net_io_start = psutil.net_io_counters()
        time_start = time_module.time()
        
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        
        # Actor side: get params and detect LoRA
        params = None
        peft_config = None
        if self._is_actor:
            params, peft_config = self._get_actor_params()
            print("TYPE OF PEFT CONFIG", type(peft_config))
            print("PEFT CONFIG for actor", peft_config)
        print("BASE SYNC DONE for actor", self.base_sync_done)

        # Rollout side: prepare vLLM model
        if self._is_rollout:
            inference_model = (
                self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
            patch_vllm_moe_model_weight_loader(inference_model)
        
        # Broadcast peft_config from actor rank 0 so rollout ranks can use LoRA path
        device = get_torch_device().current_device()
        stream = get_torch_device().current_stream()
        if torch.distributed.get_rank() == 0 and self._is_actor and peft_config is not None:
            print("TYPE OF PEFT CONFIG", type(peft_config))
            payload_bytes = json.dumps(_convert_sets_to_lists(dataclasses.asdict(peft_config))).encode("utf-8")
            length_tensor = torch.tensor([len(payload_bytes)], dtype=torch.int64, device=device)
        else:
            payload_bytes = None
            length_tensor = torch.empty(1, dtype=torch.int64, device=device)
        self._weight_sync_group.broadcast(length_tensor, src=0, stream=stream)
        payload_len = int(length_tensor.item())
        if payload_len > 0:
            buf = torch.empty(payload_len, dtype=torch.uint8, device=device)
            if torch.distributed.get_rank() == 0 and self._is_actor and payload_bytes is not None:
                buf.copy_(torch.tensor(list(payload_bytes), dtype=torch.uint8, device=device))
            self._weight_sync_group.broadcast(buf, src=0, stream=stream)
            if self._is_rollout:
                from peft.tuners.lora.config import LoraConfig
                received_json = bytes(buf.tolist()).decode("utf-8")
                peft_config = LoraConfig(**json.loads(received_json))
                self.peft_config = peft_config
            elif self._is_actor and peft_config is not None:
                self.peft_config = peft_config
        else:
            if self._is_rollout:
                peft_config = None
                self.peft_config = None

        # If this is a LoRA model and base weights are already synced, use vLLM LoRA interface
        if peft_config is not None and self.base_sync_done:
            # Sync only LoRA adapters via vLLM's add_lora
            print("Mode: LoRA") if self._is_rollout else print("Mode: Actor")
            if self._is_rollout:
                import time
                from verl.utils.vllm.utils import TensorLoRARequest
                
                # Prepare LoRA tensors
                lora_tensors = {}
                for key, shape, dtype in self._weights_info:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    if self._is_actor:
                        assert key in params
                        origin_data = params[key]
                        if hasattr(origin_data, "full_tensor"):
                            origin_data = origin_data.full_tensor()
                        if torch.distributed.get_rank() == 0:
                            tensor.copy_(origin_data)
                    
                    self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
                    
                    if self._is_rollout:
                        lora_tensors[key] = tensor
                
                # Load LoRA via vLLM
                if self._is_rollout:
                    print("Yes we do access this for LoRA")
                    lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                    lora_request = TensorLoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="verl_lora_path",
                        peft_config=dataclasses.asdict(self.peft_config),
                        lora_tensors=lora_tensors,
                    )
                    print("LoRA Request added")
                    self.rollout.inference_engine.llm_engine.add_lora(lora_request)
        else:
            # Full weight sync (first time, or non-LoRA model)
            print("Base Sync for Rollout") if self._is_rollout else print("Base Sync for Actor")
            #print("My base sync done is", self.base_sync_done)
            for key, shape, dtype in self._weights_info:
                tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                if self._is_actor:
                    assert key in params
                    origin_data = params[key]
                    if hasattr(origin_data, "full_tensor"):
                        origin_data = origin_data.full_tensor()
                    if torch.distributed.get_rank() == 0:
                        tensor.copy_(origin_data)
                
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
                if self._is_rollout:
                    inference_model.load_weights([(key, tensor)])
        
            # Mark base sync as done for actor workers
            self.base_sync_done = True
        
        # Measure network I/O after sync
        net_io_end = psutil.net_io_counters()
        time_end = time_module.time()
        
        # Calculate differences
        bytes_sent = net_io_end.bytes_sent - net_io_start.bytes_sent
        bytes_recv = net_io_end.bytes_recv - net_io_start.bytes_recv
        elapsed_time = time_end - time_start
        
        # Log network I/O statistics
        rank = torch.distributed.get_rank()
        role = "Actor" if self._is_actor else "Rollout"
        sync_type = "LoRA" if (peft_config is not None and self.base_sync_done) else "Full"
        total_bytes = bytes_sent + bytes_recv
        bandwidth_gbps = (total_bytes / (1024**3)) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"[{role} Rank {rank}] sync_rollout_weights() Network I/O:")
        print(f"  Type: {sync_type} weights sync")
        print(f"  Time: {elapsed_time:.3f}s")
        print(f"  Bytes sent: {bytes_sent / (1024**2):.2f} MB ({bytes_sent / (1024**3):.4f} GB)")
        print(f"  Bytes recv: {bytes_recv / (1024**2):.2f} MB ({bytes_recv / (1024**3):.4f} GB)")
        print(f"  Total I/O: {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.4f} GB)")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Optional: Log to CSV file for analysis
        if rank == 0:  # Only log from rank 0 to avoid file conflicts
            import csv
            import os
            log_file = "sync_weights_network_io.csv"
            file_exists = os.path.exists(log_file)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'role', 'sync_type', 'time_s', 'bytes_sent', 'bytes_recv', 'total_bytes', 'bandwidth_gbps'])
                writer.writerow([
                    time_module.time(),
                    role,
                    sync_type,
                    elapsed_time,
                    bytes_sent,
                    bytes_recv,
                    total_bytes,
                    bandwidth_gbps
                ])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        
        # Return cached info if available and still valid
        # (Note: for LoRA, info changes after base_sync_done)
        if hasattr(self, "_weights_info") and not (self._is_lora and not self.base_sync_done):
            return self._weights_info
        
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
            
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        
        params, _ = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret


class RolloutWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str):
        Worker.__init__(self)
        assert role == "rollout"
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self._is_rollout = True
        self._is_actor = False
        self.base_sync_done = False
        self.peft_config = None
        

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))

        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm"

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        from .vllm_sharding_manager import VLLMShardingManager

        rollout_sharding_manager = VLLMShardingManager(
            inference_engine=rollout.inference_engine, device_mesh=rollout_device_mesh
        )

        log_gpu_memory_usage("After building sharding manager", logger=logger)

        self.rollout = rollout
        self.rollout_sharding_manager = rollout_sharding_manager

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def async_generate_sequences(self, prompts):
        # Support all hardwares
        print("Starting async_generate_sequences")

        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        print("before rollout sharding manager")
        with self.rollout_sharding_manager:
            print("after entering rollout sharding ctx manager")
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            print("Log 1")

            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)
            print("Finish generating sequences")

            log_gpu_memory_usage("After rollout generation", logger=logger)

        print("after rollout generation")
        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
