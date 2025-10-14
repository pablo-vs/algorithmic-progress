# Setup

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation)
2. Set up the uv environment
```
uv venv
uv sync
source .venv/bin/activate
```
3. Create a [Weight and Biases](https://wandb.ai/) account if you don't have one.
4. Create a [Modal](https://modal.com) account and set it up locally. Create a new Secret in Modal called "wandb-secret" and paste your W&B API key in the default environment variable.
5. Run `python setup.py` and observe the results of the training run in your W&B dashboard.

## Model Evaluations

We monitor model performance by evaluating on a set of standard benchmarks, such as MMLU, ARC, GSM8K and Truthful Q&A. 

### Running Evaluations Locally

Evaluation tooling (`lm-eval`, HuggingFace Hub utilities) is optional.  
Install it on demand to run evals locally:

```bash
uv sync --group eval
```

Run a quick benchmark against a small HF model:

```bash
python -m eval.run_benchmarks \
  --model google/gemma-3-270m-it \
  --tasks mmlu,gsm8k,arc_easy,truthfulqa_mc2 \
  --limit 25 \
  --seed 123 \
  --device cuda:0
```

Results are written to `runs/eval_YYYYMMDD-HHMMSS.json`. You can replace the model to your own pretrained model by setting the path - note that your model has to follow HuggingFace conventions for this to work.

You might need a HuggingFace token to access gated models; add it to the .env file as ```HF_TOKEN=hf_xxx```. 

### Running Evaluations on Modal

1. Optional: store your HF token in Modal for remote jobs, if you need to access gated models:

   ```bash
   modal secret create hf-token HF_TOKEN=hf_xxx
   ```

2. Launch the remote evaluation:

   ```bash
   modal run -m eval.run_benchmarks_modal \
     --model google/gemma-3-270m-it \
     --tasks mmlu,gsm8k,arc_easy,truthfulqa_mc2 \
     --limit 25 \
     --seed 123 \
     --device cuda:0
   ```

When the job finishes the JSON results are downloaded into the local `runs/` directory. Arguments are the same as before. 

### Visualizing Evaluation Runs

Install the visualization dependencies if you have not already (they are part of the default environment), then generate a chart:

```bash
python eval/plot_benchmarks.py --run runs/eval_20251014-105528.json
```

By default the script displays the overall MMLU score and other benchmarks.  
Use `--show-mmlu-subtasks` to include per-subject bars or `--focus-prefix` / `--include-task` for custom slices.

Generated plots are saved alongside the corresponding JSON files (e.g. `runs/eval_20251014-105528.png`).

## TODO

- [ ] Define a global modal App image in a config and reuse it properly (now the `run_benchmarks_modal.py` script defines its own image which is not a best practice).
- [ ] Add basic unit tests. 
