from omegaconf import DictConfig

def test_run(cfg: DictConfig):
    import torch
    from pathlib import Path
    from litgpt.pretrain import setup
    from litgpt.data import TextFiles
    from litgpt.args import TrainArgs, EvalArgs, LogArgs
    from litgpt.scripts.download import download_from_hub

    download_from_hub(cfg.model.name, tokenizer_only=True)
    data = TextFiles(train_data_path = Path(cfg.data.train_path), seed = 42)
    train = TrainArgs(**cfg.train)
    eval_args = EvalArgs(**cfg.eval)
    log = LogArgs(**cfg.log)
    setup(cfg.model.name,
          data=data,
          train=train,
          eval=eval_args,
          log=log,
          logger_name="wandb",
          tokenizer_dir=Path(cfg.model.name)
        )

