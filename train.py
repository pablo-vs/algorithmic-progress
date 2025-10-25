from omegaconf import DictConfig

def test_run(cfg: DictConfig):
    import torch
    from pathlib import Path
    from litgpt.pretrain import setup
    from litgpt.data import TextFiles
    from litgpt.args import TrainArgs, EvalArgs, LogArgs
    from litgpt.scripts.download import download_from_hub
    import sys
    from contextlib import contextmanager
    from typing import List

    @contextmanager
    def mock_cli_args(args: List[str]):
        """Temporarily replace sys.argv with mock CLI arguments."""
        original_argv = sys.argv.copy()
        sys.argv = args
        try:
            yield
        finally:
            sys.argv = original_argv

    def dict_to_cli_args(config: dict, prefix: str = "") -> List[str]:
        """Convert nested dict to CLI argument list."""
        args = []
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                args.extend(dict_to_cli_args(value, full_key))
            else:
                args.extend([f"--{full_key}", str(value)])
        return args

    lcfg = cfg.litgpt
    download_from_hub(cfg.model_name, tokenizer_only=True)
    data = TextFiles(train_data_path = Path(cfg.data.train_path), seed = 42)
    train = TrainArgs(**lcfg.train)
    eval_args = EvalArgs(**lcfg.eval)
    log = LogArgs(**lcfg.log)

    cli_args = ["script_name.py"] + dict_to_cli_args(lcfg) + [cfg.model_name]
    with mock_cli_args(cli_args):
        setup(model_name=cfg.model_name,
              data=data,
              train=train,
              eval=eval_args,
              log=log,
              logger_name="wandb",
              tokenizer_dir=Path(cfg.model_name)
            )

