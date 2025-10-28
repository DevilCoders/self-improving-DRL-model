# Data Management and Chunking

This document covers dataset generation, chunked storage, and offline ingestion
strategies.

## Synthetic Dataset Builder

The `SyntheticDatasetBuilder` creates lightweight datasets for experimentation and
produces chunked shards alongside full-array `.npy` files.

```python
from drl_system import SystemConfig, SyntheticDatasetBuilder

config = SystemConfig()
config.dataset.chunk_size = 2048
config.dataset.chunk_overlap = 256

builder = SyntheticDatasetBuilder(config.dataset)
root = builder.generate(num_samples=10000, obs_dim=16)
```

After generation the directory layout is:

```
<root>/<version>/
  ├── observations.npy
  ├── actions.npy
  ├── rewards.npy
  └── chunks/
      ├── observations/chunk_0000.npy
      ├── actions/chunk_0000.npy
      └── rewards/chunk_0000.npy
      ...
```

## Iterating Over Chunks

Use `iter_chunks()` to stream batches without loading the entire dataset into memory.

```python
for obs_chunk, action_chunk, reward_chunk in builder.iter_chunks():
    # Feed into offline training or pre-processing pipelines
    pass
```

Chunks respect the configured `chunk_overlap`, making it easy to build sequence windows
or merge shards for meta-learning tasks.

## Offline Replay Seeding

When dataset chunks exist, the trainer automatically seeds the replay buffer with
transitions labelled with `info["source"] == "offline"`. This means that including
`"offline"` in `config.training.modes` immediately benefits from pre-generated data
before switching to online interaction.

## Best Practices

- Generate datasets per experiment version to keep metadata reproducible.
- Adjust `chunk_overlap` to support sequence context for transformers or RNN-based
  components.
- Combine synthetic data with human-collected logs by saving them in the same chunk
  schema – the iterator will consume them seamlessly.
