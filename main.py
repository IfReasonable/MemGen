import argparse
from datetime import datetime
import os
import random
import time

import numpy as np
import torch
from accelerate.utils import broadcast_object_list


def _patch_deepspeed_zero2_used_param_counter() -> None:
    """Work around DeepSpeed ZeRO1/2 crash with torch>=2.4.

    DeepSpeed's used-parameter counting can incorrectly attempt to inspect
    autograd metadata for parameters that do not require gradients, leading to:
    `AttributeError: 'NoneType' object has no attribute 'next_functions'`.

    Patch both the utils function and the ZeRO stage_1_and_2 local import.
    """

    try:
        import deepspeed.runtime.utils as ds_utils
        import deepspeed.runtime.zero.stage_1_and_2 as ds_zero
    except Exception:
        return

    if getattr(ds_utils.count_used_parameters_in_backward, "_memgen_patched", False):
        return

    # Torch 2.4+ changed internals in a way that can make
    # `torch.autograd.graph._get_grad_fn_or_grad_acc(param)` crash for some
    # parameters (it assumes `t.view_as(t).grad_fn` is always present).
    # Patch it to be defensive and return None when metadata is missing.
    try:
        import torch.autograd.graph as torch_ag

        if not getattr(torch_ag._get_grad_fn_or_grad_acc, "_memgen_patched", False):
            orig_get = torch_ag._get_grad_fn_or_grad_acc

            def _safe_get_grad_fn_or_grad_acc(t):
                if not getattr(t, "requires_grad", False):
                    return None
                gf = getattr(t, "grad_fn", None)
                if gf is not None:
                    return gf

                # Leaf tensor case: try to retrieve AccumulateGrad via view.
                try:
                    v = t.view_as(t)
                    v_gf = getattr(v, "grad_fn", None)
                    if v_gf is not None and getattr(v_gf, "next_functions", None):
                        return v_gf.next_functions[0][0]
                except Exception:
                    pass

                # Fallbacks for edge cases where view_as() produces no grad_fn.
                for op in (
                    lambda x: x.expand_as(x),
                    lambda x: x.reshape(x.shape),
                    lambda x: x + 0,
                ):
                    try:
                        y = op(t)
                        y_gf = getattr(y, "grad_fn", None)
                        if y_gf is not None and getattr(y_gf, "next_functions", None):
                            return y_gf.next_functions[0][0]
                    except Exception:
                        continue

                # Give up: treat as not participating in autograd.
                return None

            _safe_get_grad_fn_or_grad_acc._memgen_patched = True  # type: ignore[attr-defined]
            torch_ag._get_grad_fn_or_grad_acc = _safe_get_grad_fn_or_grad_acc

        # Also patch DeepSpeed's local reference if present.
        if hasattr(ds_utils, "_get_grad_fn_or_grad_acc"):
            ds_utils._get_grad_fn_or_grad_acc = torch_ag._get_grad_fn_or_grad_acc
    except Exception:
        pass

    orig_fn = ds_utils.count_used_parameters_in_backward

    def wrapped(params, *args, **kwargs):
        filtered = [p for p in params if getattr(p, "requires_grad", False)]
        return orig_fn(filtered, *args, **kwargs)

    wrapped._memgen_patched = True  # type: ignore[attr-defined]
    ds_utils.count_used_parameters_in_backward = wrapped
    if hasattr(ds_zero, "count_used_parameters_in_backward"):
        ds_zero.count_used_parameters_in_backward = wrapped

from common.config import Config
from common.logger import setup_logger
from data import get_data_builder
from memgen.model import MemGenModel
from memgen.runner import MemGenRunner

def set_seed(random_seed: int, use_gpu: bool):

    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

    print(f"set seed: {random_seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Generator")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def build_working_dir(config: Config) -> str:
    
    # parent dir: <train/evaluate>/<dataset_name>/<reasoner_model_name>
    mode = config.run_cfg.mode
    dataset_name = config.dataset_cfg.name
    model_name = config.model_cfg.model_name.split("/")[1]
    parent_dir = os.path.join("results", mode, dataset_name, model_name)

    # Ensure the parent directory exists before any multi-process rendezvous.
    os.makedirs(parent_dir, exist_ok=True)

    # name: <prompt_aug_num>_<prompt_latents_len>_<inference_aug_num>_<inference_latents_len>_<timestamp>
    max_prompt_aug_num = config.model_cfg.max_prompt_aug_num
    prompt_latents_len = config.model_cfg.weaver.prompt_latents_len
    max_inference_aug_num = config.model_cfg.max_inference_aug_num
    inference_latents_len = config.model_cfg.weaver.inference_latents_len
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return default

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)

    # NOTE: build_working_dir() runs BEFORE accelerate/deepspeed initializes torch.distributed.
    # Relying solely on torch.distributed.is_initialized() can therefore cause each rank to
    # generate its own timestamp -> multiple work dirs. Use an env-based rendezvous instead.
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if world_size > 1:
        coord_key = os.environ.get("MASTER_PORT") or os.environ.get("MAIN_PROCESS_PORT") or "default"
        coord_file = os.path.join(parent_dir, f".working_dir_{coord_key}.txt")

        if rank == 0:
            # Try to create the coordination file exclusively. If it's stale from a previous run
            # with the same port, overwrite it.
            try:
                fd = os.open(coord_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w") as f:
                    f.write(run_time)
            except FileExistsError:
                try:
                    mtime = os.path.getmtime(coord_file)
                    # 10 minutes should be enough for normal usage; treat older as stale.
                    if time.time() - mtime > 600:
                        os.remove(coord_file)
                        fd = os.open(coord_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                        with os.fdopen(fd, "w") as f:
                            f.write(run_time)
                    else:
                        with open(coord_file, "r") as f:
                            run_time = (f.read() or run_time).strip()
                except Exception:
                    # As a fallback, keep local run_time.
                    pass
        else:
            # Wait for rank0 to write the coordination file.
            deadline = time.time() + 120  # seconds
            while time.time() < deadline and not os.path.exists(coord_file):
                time.sleep(0.1)
            try:
                with open(coord_file, "r") as f:
                    run_time = (f.read() or run_time).strip()
            except Exception:
                pass

    # Secondary path: if torch.distributed is already initialized (e.g., some launchers),
    # broadcast again to be safe.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        obj_list = [run_time] if torch.distributed.get_rank() == 0 else [None]
        broadcast_object_list(obj_list, from_process=0)
        run_time = obj_list[0]
    working_dir = f"pn={max_prompt_aug_num}_pl={prompt_latents_len}_in={max_inference_aug_num}_il={inference_latents_len}_{run_time}" 

    return os.path.join(parent_dir, working_dir)

def main():

    args = parse_args()
    config = Config(args)

    # Must run before accelerate/deepspeed engine is created.
    _patch_deepspeed_zero2_used_param_counter()

    # Enforce strict loading rules for full MemGen evaluation and trigger training.
    eval_vanilla = bool(getattr(config.run_cfg, "eval_vanilla", False))
    if config.run_cfg.mode == "evaluate" and not eval_vanilla:
        # Default behavior: evaluation expects a full MemGen checkpoint (weaver+trigger extras).
        config.model_cfg.require_full_memgen = True
        config.model_cfg.strict_load_weaver = True
        config.model_cfg.strict_load_trigger = True
        config.model_cfg.trigger.active = True
    elif config.run_cfg.mode == "train" and config.run_cfg.train_trigger:
        config.model_cfg.strict_load_weaver = True

    set_seed(config.run_cfg.seed, use_gpu=True)
    
    # set up working directory
    working_dir = build_working_dir(config)
    
    # set up logger
    config.run_cfg.log_dir = os.path.join(working_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)

    config.pretty_print()

    # build components
    config_dict = config.to_dict()
    data_builder = get_data_builder(config_dict.get("dataset"))
    model = MemGenModel.from_config(config_dict.get("model"))
    
    runner = MemGenRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=working_dir
    )

    # train or evaluate 
    if config.run_cfg.mode == "train":
        runner.train()
    
    elif config.run_cfg.mode == "evaluate":
        runner.evaluate()

    # Cleanly tear down distributed process group (PyTorch 2.4+ warns otherwise).
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass

if __name__ == "__main__":
    main()