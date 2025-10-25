import modal
import hydra
from omegaconf import DictConfig, OmegaConf 
from train import test_run

app = modal.App("test-spar-stack")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.56.1", 
        "accelerate==1.10.1",
        "litgpt[extra]==0.5.11",
        "omegaconf",
        "wandb"
    ).add_local_dir("data/train", remote_path="/data")
    .add_local_python_source("train")
)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    test_run_modal = app.function(
            image=image,
            secrets=[modal.Secret.from_name("wandb-secret")],
            **cfg.modal
        )(test_run)

    with modal.enable_output():
        with app.run():
            test_run_modal.remote(cfg)

if __name__ == "__main__":
    main()
