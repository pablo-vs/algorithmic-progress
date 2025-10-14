# eval/run_benchmarks_modal.py
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import modal
from modal import gpu as modal_gpu
from omegaconf import OmegaConf

from eval.run_benchmarks import run_eval, save_results, DEFAULT_TASKS

app = modal.App("algo-progress-eval")

# Load default Modal settings from the project config.
_CONFIG_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "conf" / "config.yaml",
    Path("/workspace/conf/config.yaml"),
]
for _cfg_path in _CONFIG_CANDIDATES:
    if _cfg_path.exists():
        CONFIG_PATH = _cfg_path
        break
else:
    raise FileNotFoundError(
        f"Could not locate config.yaml. Checked: {', '.join(str(p) for p in _CONFIG_CANDIDATES)}"
    )

CONFIG = OmegaConf.load(CONFIG_PATH)
MODAL_SETTINGS_RAW = OmegaConf.to_container(CONFIG.modal, resolve=True, enum_to_str=True)  # type: ignore[arg-type]
if not isinstance(MODAL_SETTINGS_RAW, dict):
    raise TypeError("Expected `modal` section in config to convert to a dictionary.")
MODAL_SETTINGS: Dict[str, Any] = MODAL_SETTINGS_RAW
def resolve_gpu_spec(spec: Any):
    if spec is None:
        return "A10G"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        type_name = spec.get("type")
        if type_name is None:
            raise ValueError("GPU spec dict must include a 'type' key.")
        attr_name = str(type_name).upper()
        if not hasattr(modal_gpu, attr_name):
            raise ValueError(f"Unknown GPU type '{type_name}'.")
        attr = getattr(modal_gpu, attr_name)
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return attr(**kwargs)
    return spec

MODAL_GPU = resolve_gpu_spec(MODAL_SETTINGS.get("gpu"))
MODAL_TIMEOUT = MODAL_SETTINGS.get("timeout", 1800)
try:
    MODAL_TIMEOUT = int(MODAL_TIMEOUT)
except (TypeError, ValueError):
    raise ValueError("`modal.timeout` must be an integer (seconds).")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "tokenizers>=0.19",
        "transformers>=4.56",
        "accelerate>=1.0",
        "python-dotenv>=1.0",
        "huggingface_hub>=0.24",
        "lm-eval==0.4.9",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
    )
    .add_local_dir(".", remote_path="/workspace")
)

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT,
    secrets=[modal.Secret.from_name("hf-token")],
)
def _run_remote(
    model: str,
    tasks: list[str],
    limit: Optional[int],
    seed: int,
    batch_size: Optional[int],
    device: Optional[str],
    extra_model_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # HF_TOKEN comes from the Modal secret; run_eval will pick it up via login()
    os.chdir("/workspace")
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
    return run_eval(
        model_name=model,
        tasks=tasks,
        limit=limit,
        seed=seed,
        batch_size=batch_size,
        device=device or "cuda:0",
        extra_model_args=extra_model_args,
    )

@app.local_entrypoint()
def main(
    model: str,
    tasks: str = ",".join(DEFAULT_TASKS),
    limit: int | None = 25,
    seed: int = 123,
    batch_size: int | None = None,
    device: str | None = None,
    extra_model_args: str | None = None,
    output: str | None = None,
) -> None:
    """
    Launch the evaluation on Modal and save the JSON result locally (runs/eval_*.json).
    Use `modal run -m eval.run_benchmarks_modal --model ...` to kick it off.
    """
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    extra = {}
    if extra_model_args:
        for kv in extra_model_args.split(","):
            k, v = kv.split("=", 1)
            extra[k] = v

    results = _run_remote.remote(
        model=model,
        tasks=task_list,
        limit=limit,
        seed=seed,
        batch_size=batch_size,
        device=device,
        extra_model_args=extra or None,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = output or f"runs/eval_{timestamp}.json"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    save_results(results, out_path)
    print(f"[modal-eval] Saved results to {out_path}")
