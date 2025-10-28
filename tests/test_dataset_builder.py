from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from drl_system.config import DatasetConfig
from drl_system.data.dataset_builder import SyntheticDatasetBuilder


def test_dataset_builder_generates(tmp_path: Path):
    config = DatasetConfig(root=str(tmp_path), version="test")
    builder = SyntheticDatasetBuilder(config)
    root = builder.generate(num_samples=10, obs_dim=2)
    obs, actions, rewards = builder.load()
    assert obs.shape == (10, 2)
    assert actions.shape[0] == 10
    assert rewards.shape[0] == 10
    assert root.exists()
    multimodal_root = root / "multimodal"
    assert multimodal_root.exists()
    # Ensure metadata files exist for configured datasets
    metadata_files = list(multimodal_root.glob("*/metadata.json"))
    assert metadata_files, "Expected multimodal dataset metadata files to be generated"


def test_dataset_builder_chunking(tmp_path: Path):
    config = DatasetConfig(root=str(tmp_path), version="chunked", chunk_size=4, chunk_overlap=2)
    builder = SyntheticDatasetBuilder(config)
    builder.generate(num_samples=10, obs_dim=2)
    chunks = list(builder.iter_chunks())
    # Overlap ensures more than two chunks for 10 samples when chunk size is 4 and overlap 2
    assert len(chunks) >= 3
    for obs_chunk, action_chunk, reward_chunk in chunks:
        assert obs_chunk.shape[0] <= 4
        assert action_chunk.shape[0] == obs_chunk.shape[0]
        assert reward_chunk.shape[0] == obs_chunk.shape[0]


def test_multimodal_datasets_materialised(tmp_path: Path):
    config = DatasetConfig(root=str(tmp_path), version="rich")
    builder = SyntheticDatasetBuilder(config)
    root = builder.generate(num_samples=4, obs_dim=3)
    multimodal_root = root / "multimodal"
    # Terminal dataset checks
    terminal_dataset = multimodal_root / "terminal_commands"
    assert (terminal_dataset / "dataset.csv").exists()
    assert (terminal_dataset / "dataset.tsv").exists()
    # Stable diffusion image artefacts
    diffusion = multimodal_root / "stable_diffusion_prompts" / "images"
    assert any(diffusion.glob("*.png"))
    # Audio dataset contains wav files and transcripts
    audio = multimodal_root / "audio_language_corpus"
    assert any((audio / "audio").glob("*.wav"))
    assert (audio / "transcripts.jsonl").exists()
    # PDF dataset contains pdf outputs
    pdfs = multimodal_root / "technical_pdfs"
    assert any((pdfs / "pdfs").glob("*.pdf"))
    # Code dataset contains multi-language snippets
    code = multimodal_root / "code_corpus"
    assert (code / "python" / "analysis_agent.py").exists()
    assert (code / "cpp" / "control_loop.cpp").exists()
    # Robotics dataset contains control traces
    robotics = multimodal_root / "robotics_controls"
    assert (robotics / "trajectories.csv").exists()
    assert (robotics / "ros_topics.json").exists()
