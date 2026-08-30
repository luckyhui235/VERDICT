# VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction

> **News**
>
> **[2026]** VERDICT is publicly available on [arXiv](https://arxiv.org/abs/2603.19306) and has been accepted to **Findings of EMNLP 2026**.

VERDICT is a self-refining multi-agent framework for legal judgment prediction (LJP). It organizes legal reasoning as a traceable virtual collegial panel: specialized agents structure the facts, retrieve legal authorities, draft a judgment, verify it, and revise it before a final decision is produced.

[[Paper]](https://arxiv.org/abs/2603.19306)

The framework is designed for research on legally grounded and interpretable prediction of applicable law articles, charges, and terms of penalty. It is **not** intended to provide legal advice or to support real-world judicial decision-making without qualified human review.

## Overview

Legal judgment prediction requires more than matching a case description to a label. A reliable prediction should connect legally salient facts to statutory elements, distinguish nearby legal concepts, and remain robust as adjudicative practice evolves.

VERDICT addresses this problem with two complementary components:

- **Directive-Informed Collegial Teams.** Five specialized roles simulate a virtual collegial panel: a Court Clerk extracts factual elements; a Judicial Assistant retrieves and re-ranks candidate statutes; a Case-handling Judge drafts a grounded opinion; an Adjudication Supervisor checks it and returns explicit `Pass`/`Reject` feedback; and a Presiding Judge consolidates the verified verdict.
- **Hybrid Jurisprudential Memory (HJM).** The memory combines precedent standards with evolving, natural-language Micro-Directives. Verified trajectories are distilled into directives that capture decision boundaries, helping the system reuse validated experience across cases.

Together, these components form a transparent **draft → verify → revise** workflow. The output is a final prediction accompanied by intermediate reasoning artifacts and revision feedback.

## Highlights

- A traceable multi-agent workflow for fact-to-element legal reasoning.
- Explicit supervisory verification with corrective feedback and iterative revision.
- Hybrid Jurisprudential Memory for accumulating validated precedent-oriented experience.
- Evaluation on CAIL2018 and CJO2025, a strict future time-split benchmark constructed from judgments after 1 January 2025.

## Paper Results

The paper reports the following Law Article prediction results:

| Dataset | VERDICT Acc. | VERDICT Macro-F1 | Previous strongest reported Acc. |
| --- | ---: | ---: | ---: |
| CAIL2018 | 85.35 | 83.29 | 83.21 (PLJP) |
| CJO2025 | 90.56 | 87.94 | 86.52 (G-Memory) |

Please refer to the paper for the full results on law articles, charges, and terms of penalty, as well as the ablations and backbone-robustness experiments.

## Repository Structure

```text
.
├── mas/                         # LLM interfaces, reasoning modules, agents, and memory
│   ├── agents/                  # Agent abstractions
│   ├── memory/mas_memory/       # Case memory and evolving memory implementation
│   └── reasoning/               # Reasoning module interfaces
├── legal-task/                  # LJP task runner and evaluation environment
│   ├── envs/                    # CAIL2018 environment and evaluator
│   ├── mas_workflow/macnet/     # Multi-agent communication workflow
│   ├── retrieval/               # Statute retrieval
│   └── run.py                   # Main entry point
├── law_article/                 # Statute mapping used by the retriever
├── data/                        # Benchmark files (see the Data section)
├── configs/                     # Generation configuration
├── requirements.txt
└── run_mas.sh                   # Convenience script for background evaluation
```

## Installation

The codebase has been developed with Python 3.12.

```bash
conda create -n verdict python=3.12
conda activate verdict

pip install -r requirements.txt
```

The default embedding configuration in [`legal-task/configs.yaml`](legal-task/configs.yaml) uses the `Qwen/Qwen3-Embedding-8B` model identifier. Set `EMBEDDING_MODEL` to a local checkpoint path or another available model identifier before running an experiment.

## Model Configuration

VERDICT supports OpenAI-compatible APIs and local Qwen-family models.

For DeepSeek's OpenAI-compatible API, export a key before running:

```bash
export DEEPSEEK_API_KEY="<your-key>"
```

For another OpenAI-compatible provider, configure both variables:

```bash
export OPENAI_API_BASE="https://<provider-endpoint>/v1"
export OPENAI_API_KEY="<your-key>"
```

Do not commit keys, provider credentials, model paths, or local data paths. Keep them in an ignored local `.env` file or in your shell environment.

You may copy [`.env.example`](.env.example) to `.env` and fill only the variables required by your setup. The following optional variables keep local paths and run outputs out of the tracked configuration:

```bash
EMBEDDING_MODEL="/path/to/embedding-model"
TERM_BUCKET_MAPPING_PATH="/path/to/time2id.json"
EVAL_OUTPUT_DIR="/path/to/evaluation-output"
```

## Running Evaluation

The current task entry point exposes the CAIL2018 evaluation workflow. A foreground run can be started with:

```bash
python legal-task/run.py \
  --task cail2018 \
  --reasoning io \
  --mas_memory case-memory \
  --max_trials 6 \
  --mode test \
  --mas_type macnet \
  --model deepseek-chat \
  --successful_topk 2 \
  --threshold 0.3
```

Alternatively, use the provided convenience script, which runs the job in the background and writes logs under `logs/`:

```bash
bash run_mas.sh
```

For local Qwen inference, pass a local checkpoint path or a Qwen model identifier to `--model`. GPU and batching for the embedding model can be configured through environment variables:

```bash
export EMBEDDING_DEVICE="cuda:0"
export EMBEDDING_BATCH_SIZE=16
```

Run artifacts, evaluation predictions, and persistent memory are written beneath `.db/`, organized by the LLM, embedding model, task, workflow, and memory configuration.

## Data

The paper evaluates VERDICT on the following datasets:

- **CAIL2018 (CAIL-Small):** the primary benchmark for legal judgment prediction. The repository expects its evaluation data under `data/cail2018/`. If you use the optional term-bucket mapping, place it at `data/cail2018/time2id.json` or set `TERM_BUCKET_MAPPING_PATH`.
- **CJO2025:** a future time-split benchmark constructed from China Judgments Online judgments after 1 January 2025. It is used to evaluate temporal generalization and reduce potential contamination by the pre-training corpora of the evaluated LLMs.
- **Statutory library:** `law_article/law_articles_mapping.jsonl` supplies the statute text used during retrieval.

Users are responsible for complying with the terms of the original data providers, applicable laws, and any privacy or redistribution restrictions. If a dataset cannot be redistributed under its original terms, obtain it from the original source and place it in the documented path rather than uploading it to an issue or pull request.

## Reproducibility Notes

- The paper uses a heterogeneous setup: a legally aligned Qwen2.5-7B-Instruct expert model for the Case-handling Judge and DeepSeek-V3 for auxiliary agents.
- Results depend on the model provider, prompt implementation, embedding model, retrieval settings, and data preprocessing. Record the exact model versions and configuration files for each run.
- The runner currently supports `cail2018`; release the corresponding task configuration and preprocessing instructions before claiming a fully reproducible CJO2025 command-line benchmark.

## Citation

If you find VERDICT useful, please cite the paper:

```bibtex
@article{liao2026verdict,
  title   = {VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction},
  author  = {Liao, Hui and Qin, Chuan and Ren, Yongwen and Li, Hao and Huang, Zhenya and Zhang, Yanyong and Wang, Chao},
  journal = {arXiv preprint arXiv:2603.19306},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.19306}
}
```

## Acknowledgements

This project builds on open-source libraries including PyTorch, Transformers, LangChain, Chroma, and Sentence Transformers. We also thank the creators and maintainers of the CAIL benchmark and China Judgments Online for making legal-reasoning research possible.

## License

A license has not yet been added to this repository. Please add a license file before making the repository public; without one, others do not receive permission to reuse, modify, or distribute the code.
