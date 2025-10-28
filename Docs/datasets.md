# Dataset Catalogue

The dataset factory can materialise multiple modality-specific corpora. Each dataset is
chunked to `DatasetConfig.chunk_size` with `chunk_overlap` controls, enabling efficient
streaming for both CPU and GPU workloads.

## Available Datasets

| Folder | Formats | Description |
| --- | --- | --- |
| `terminal_commands/` | `csv`, `tsv`, `txt`, `json`, `jsonl` | Cross-platform automation snippets covering Linux/Windows, admin and standard roles. |
| `security_commands/` | `csv`, `tsv`, `txt`, `json`, `jsonl` | Safety-filtered offensive/defensive playbooks for ethical hacking workflows. |
| `stable_diffusion/` | `png`, `json` | Image prompts, seeds, and latent metadata for Stable Diffusion fine-tuning. |
| `audio_language/` | `wav`, `jsonl` | Speech corpora with transcripts for NLP/NLU/NLG scenarios. |
| `pdf_knowledge/` | `pdf`, `json` | Portable document primers with extracted structure metadata. |
| `code_corpus/` | `py`, `cpp`, `js`, `json` | Polyglot programming references for grounded reasoning. |
| `robotics_controls/` | `json`, `csv` | Trajectory sketches, ROS topic annotations, and actuator envelopes for real devices. |

## Chunking Workflow

1. Invoke `SyntheticDatasetBuilder.build_all()`.
2. Each dataset writes raw artifacts under `data/generated/<version>/<dataset_name>/`.
3. The builder then slices the data into chunks under `.../chunks/<dataset>/<chunk_id>.npz`
   with overlap to maintain temporal context for sequence models.
4. `builder.iter_chunks()` yields `(observations, actions, rewards)` arrays ready to seed
   the replay buffer for offline or warm-start runs.

## Extending Datasets

- Append new `DatasetSpec` entries in `drl_system/config.py` to register additional
  datasets. Provide formats, tags, and sample counts.
- Implement custom generators in `drl_system/data/dataset_builder.py`—the builder already
  contains hooks for commands, diffusion prompts, audio snippets, PDF synthesis, and
  code generation.
- Use `builder.export_manifest()` to produce dataset manifests for compliance tracking.

Refer to `docs/data_management.md` for operational policies and retention guidelines.
