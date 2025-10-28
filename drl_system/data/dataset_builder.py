"""Synthetic dataset builder for offline experimentation."""
from __future__ import annotations

import csv
import json
import math
import random
import wave
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image

from ..config import DatasetConfig, DatasetSpec


@dataclass
class SyntheticDatasetBuilder:
    config: DatasetConfig

    def generate(self, num_samples: int = 1000, obs_dim: int = 8) -> Path:
        root = Path(self.config.root) / self.config.version
        root.mkdir(parents=True, exist_ok=True)
        observations = np.random.randn(num_samples, obs_dim).astype("float32")
        actions = np.random.randint(0, 4, size=(num_samples, 1)).astype("int64")
        rewards = np.random.randn(num_samples, 1).astype("float32")
        np.save(root / "observations.npy", observations)
        np.save(root / "actions.npy", actions)
        np.save(root / "rewards.npy", rewards)
        self._materialize_chunks(root, observations, actions, rewards)
        self._generate_multimodal_datasets(root)
        return root

    def load(self) -> List[np.ndarray]:
        root = Path(self.config.root) / self.config.version
        return [np.load(root / name) for name in ["observations.npy", "actions.npy", "rewards.npy"]]

    # Chunking -------------------------------------------------------------
    def _chunk_array(self, array: np.ndarray) -> List[np.ndarray]:
        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, min(self.config.chunk_overlap, chunk_size - 1))
        chunks: List[np.ndarray] = []
        start = 0
        while start < array.shape[0]:
            end = min(start + chunk_size, array.shape[0])
            chunks.append(array[start:end])
            if end == array.shape[0]:
                break
            start = end - overlap
        return chunks

    def _materialize_chunks(
        self,
        root: Path,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        chunk_root = root / "chunks"
        (chunk_root / "observations").mkdir(parents=True, exist_ok=True)
        (chunk_root / "actions").mkdir(parents=True, exist_ok=True)
        (chunk_root / "rewards").mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(self._chunk_array(observations)):
            np.save(chunk_root / "observations" / f"chunk_{idx:04d}.npy", chunk)
        for idx, chunk in enumerate(self._chunk_array(actions)):
            np.save(chunk_root / "actions" / f"chunk_{idx:04d}.npy", chunk)
        for idx, chunk in enumerate(self._chunk_array(rewards)):
            np.save(chunk_root / "rewards" / f"chunk_{idx:04d}.npy", chunk)

    def iter_chunks(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        root = Path(self.config.root) / self.config.version / "chunks"
        observation_files = sorted((root / "observations").glob("chunk_*.npy"))
        action_files = sorted((root / "actions").glob("chunk_*.npy"))
        reward_files = sorted((root / "rewards").glob("chunk_*.npy"))
        for obs_file, act_file, rew_file in zip(observation_files, action_files, reward_files):
            yield np.load(obs_file), np.load(act_file), np.load(rew_file)

    # Multimodal dataset construction ------------------------------------
    def _generate_multimodal_datasets(self, root: Path) -> None:
        multi_root = root / "multimodal"
        multi_root.mkdir(parents=True, exist_ok=True)
        for spec in self.config.datasets:
            dataset_root = multi_root / spec.name
            dataset_root.mkdir(parents=True, exist_ok=True)
            metadata = {
                "name": spec.name,
                "category": spec.category,
                "modality": spec.modality,
                "formats": spec.formats,
                "samples": spec.samples,
                "description": spec.description,
                "tags": spec.tags or [],
            }
            (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
            handler = self._dataset_handlers().get(spec.category, self._build_structured_text_dataset)
            handler(dataset_root, spec)

    def _dataset_handlers(self) -> Dict[str, Callable[[Path, DatasetSpec], None]]:
        return {
            "terminal_commands": self._build_terminal_dataset,
            "security_commands": self._build_security_dataset,
            "stable_diffusion": self._build_stable_diffusion_dataset,
            "audio_language": self._build_audio_language_dataset,
            "pdf_knowledge": self._build_pdf_dataset,
            "code_corpus": self._build_code_dataset,
            "robotics_controls": self._build_robotics_dataset,
        }

    def _build_structured_text_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        entries = [
            {
                "command": f"echo Placeholder command {idx}",
                "platform": "generic",
                "requires_admin": False,
                "description": "Placeholder command entry",
            }
            for idx in range(spec.samples)
        ]
        self._write_structured_text_formats(dataset_root, spec, entries)

    def _build_terminal_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        linux_admin = [
            ("linux", True, "sudo apt update", "Update package repository cache"),
            ("linux", True, "sudo systemctl restart networking", "Restart network services"),
            ("linux", False, "htop", "Inspect system processes interactively"),
            ("linux", False, "journalctl --user", "Read user scoped journal entries"),
        ]
        windows_admin = [
            ("windows", True, "Get-WindowsUpdate", "Query pending Windows updates"),
            ("windows", True, "Restart-Service Spooler", "Restart the print spooler service"),
            ("windows", False, "Get-Process", "List active processes"),
            ("windows", False, "Test-NetConnection", "Validate port reachability"),
        ]
        automation_sequences = [
            ("linux", False, "crontab -l", "Inspect scheduled automation tasks"),
            ("windows", False, "schtasks /Query /FO LIST", "Enumerate task scheduler jobs"),
        ]
        records = linux_admin + windows_admin + automation_sequences
        while len(records) < spec.samples:
            base = random.choice(records)
            records.append((base[0], base[1], base[2], base[3]))
        entries = [
            {
                "platform": platform,
                "requires_admin": requires_admin,
                "command": command,
                "description": description,
                "category": "automation" if "task" in command.lower() else "operations",
            }
            for platform, requires_admin, command, description in records[: spec.samples]
        ]
        self._write_structured_text_formats(dataset_root, spec, entries)

    def _build_security_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        safe_playbook = [
            {
                "phase": "recon",
                "command": "nmap -sV --script safe 10.0.0.0/24",
                "description": "Safe service enumeration with default scripts",
                "mitigation": "Review authorised scope before execution",
            },
            {
                "phase": "vuln-scan",
                "command": "nikto -h https://target.example",
                "description": "Scan for outdated web server components",
                "mitigation": "Run only against assets with explicit approval",
            },
            {
                "phase": "posture",
                "command": "lynis audit system",
                "description": "System hardening audit",
                "mitigation": "Capture baseline before applying remediations",
            },
            {
                "phase": "reporting",
                "command": "curl https://haveibeenpwned.com/api/v3/breachedaccount/{email}",
                "description": "Check exposure of corporate accounts",
                "mitigation": "Use API keys with logging enabled",
            },
        ]
        while len(safe_playbook) < spec.samples:
            template = random.choice(safe_playbook)
            safe_playbook.append(dict(template))
        entries = safe_playbook[: spec.samples]
        self._write_structured_text_formats(dataset_root, spec, entries)

    def _write_structured_text_formats(
        self, dataset_root: Path, spec: DatasetSpec, entries: List[Dict[str, object]]
    ) -> None:
        fieldnames = sorted({key for entry in entries for key in entry.keys()})
        if "csv" in spec.formats:
            with (dataset_root / "dataset.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(entries)
        if "tsv" in spec.formats:
            with (dataset_root / "dataset.tsv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)
        if "txt" in spec.formats:
            lines = [f"{entry.get('command', entry)} -- {entry.get('description', '')}" for entry in entries]
            (dataset_root / "dataset.txt").write_text("\n".join(lines))
        if "json" in spec.formats:
            (dataset_root / "dataset.json").write_text(json.dumps(entries, indent=2))
        if "jsonl" in spec.formats:
            with (dataset_root / "dataset.jsonl").open("w") as fh:
                for entry in entries:
                    fh.write(json.dumps(entry) + "\n")

    def _build_stable_diffusion_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        images_root = dataset_root / "images"
        images_root.mkdir(exist_ok=True)
        prompts: List[Dict[str, object]] = []
        for idx in range(spec.samples):
            prompt = random.choice(
                [
                    "sunset over a futuristic cityscape",
                    "macro photograph of dew on a leaf",
                    "diagrammatic render of a robotic arm",
                    "surreal landscape of floating islands",
                    "architectural concept art for eco home",
                ]
            )
            guidance = random.uniform(6.0, 12.0)
            seed = random.randint(0, 2**32 - 1)
            noise = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            image = Image.fromarray(noise, mode="RGB")
            image_path = images_root / f"sample_{idx:04d}.png"
            image.save(image_path)
            prompts.append(
                {
                    "prompt": prompt,
                    "guidance": round(guidance, 2),
                    "seed": seed,
                    "image": image_path.name,
                }
            )
        (dataset_root / "prompts.json").write_text(json.dumps(prompts, indent=2))

    def _build_audio_language_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        audio_root = dataset_root / "audio"
        audio_root.mkdir(exist_ok=True)
        transcripts: List[Dict[str, object]] = []
        sample_rate = 16_000
        durations = [0.75, 1.0, 1.25, 1.5]
        phrases = [
            ("en", "Policy gradient agents coordinate across clusters."),
            ("ru", "Агент изучает новые стратегии безопасно."),
            ("es", "El sistema adapta su comportamiento con retroalimentación."),
            ("de", "Das Modell nutzt Meta-Lernen für Verbesserungen."),
        ]
        for idx in range(spec.samples):
            lang, text = random.choice(phrases)
            freq = random.choice([220.0, 320.0, 440.0, 550.0])
            duration = random.choice(durations)
            num_samples = int(sample_rate * duration)
            times = np.linspace(0.0, duration, num=num_samples, endpoint=False)
            waveform = (0.3 * np.sin(2 * math.pi * freq * times)).astype(np.float32)
            scaled = np.int16(waveform * 32767)
            wav_path = audio_root / f"utterance_{idx:04d}.wav"
            with closing(wave.open(str(wav_path), "w")) as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(scaled.tobytes())
            transcripts.append({"audio": wav_path.name, "language": lang, "text": text, "duration": duration})
        with (dataset_root / "transcripts.jsonl").open("w") as fh:
            for entry in transcripts:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _build_pdf_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        pdf_root = dataset_root / "pdfs"
        pdf_root.mkdir(exist_ok=True)
        summaries = []
        topics = [
            "Safe exploration policies",
            "Meta-learning loops",
            "Distributed rollout orchestration",
            "Human feedback integration",
            "Hardware deployment playbooks",
        ]
        for idx in range(spec.samples):
            topic = topics[idx % len(topics)]
            text = f"{topic}\n\nThis briefing summarises latest best practices for {topic.lower()}."
            pdf_path = pdf_root / f"briefing_{idx:04d}.pdf"
            self._write_minimal_pdf(pdf_path, text)
            summaries.append({"pdf": pdf_path.name, "topic": topic})
        (dataset_root / "summaries.json").write_text(json.dumps(summaries, indent=2))

    def _write_minimal_pdf(self, path: Path, text: str) -> None:
        safe_text = text.replace("(", "[").replace(")", "]")
        stream = f"BT /F1 12 Tf 50 780 Td ({safe_text}) Tj ET".encode("utf-8")
        objects = [
            b"1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj\n",
            b"2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj\n",
            b"3 0 obj<</Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>endobj\n",
            b"4 0 obj<</Length "
            + str(len(stream)).encode("ascii")
            + b" >>stream\n"
            + stream
            + b"\nendstream endobj\n",
            b"5 0 obj<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj\n",
        ]
        header = b"%PDF-1.4\n"
        body = b""
        offsets: List[int] = []
        for obj in objects:
            offsets.append(len(header) + len(body))
            body += obj
        xref = [b"xref\n0 6\n0000000000 65535 f \n"]
        for offset in offsets:
            xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
        xref_bytes = b"".join(xref)
        trailer = (
            b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n"
            + str(len(header) + len(body)).encode("ascii")
            + b"\n%%EOF"
        )
        path.write_bytes(header + body + xref_bytes + trailer)

    def _build_code_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        languages = {
            "python": (
                "analysis_agent.py",
                """def plan(actions: list[str]) -> None:\n    for step, action in enumerate(actions, start=1):\n        print(f\"[PLAN] {step}: {action}\")\n\n\nif __name__ == \"__main__\":\n    plan([\"collect_rollout\", \"update_policy\", \"evaluate\"])\n""",
            ),
            "cpp": (
                "control_loop.cpp",
                """#include <iostream>\n#include <vector>\n\nint main() {\n    std::vector<double> rewards{1.0, 0.4, -0.1};\n    double baseline = 0.0;\n    for (const auto &reward : rewards) {\n        baseline += reward;\n    }\n    std::cout << \"Average reward: \" << baseline / rewards.size() << std::endl;\n    return 0;\n}\n""",
            ),
            "javascript": (
                "dashboard.js",
                """export function renderStatus(metrics) {\n  return Object.entries(metrics)\n    .map(([key, value]) => `${key}: ${value.toFixed(3)}`)\n    .join('\n');\n}\n""",
            ),
        }
        for language, (filename, content) in languages.items():
            lang_root = dataset_root / language
            lang_root.mkdir(exist_ok=True)
            (lang_root / filename).write_text(content)
        snippets = [
            {
                "language": language,
                "file": filename,
                "description": "Code sample generated for multimodal grounding",
            }
            for language, (filename, _) in languages.items()
        ]
        (dataset_root / "snippets.json").write_text(json.dumps(snippets, indent=2))

    def _build_robotics_dataset(self, dataset_root: Path, spec: DatasetSpec) -> None:
        fieldnames = [
            "trajectory_id",
            "ros_topic",
            "actuator",
            "target_position",
            "velocity_limit",
            "safety_margin",
        ]
        rows = []
        for idx in range(spec.samples):
            ros_topic = random.choice(["/arm/joint_states", "/base/velocity", "/gripper/command"])
            actuator = random.choice(["shoulder_pan", "shoulder_lift", "wrist_roll", "wheel", "gripper"])
            target_position = round(random.uniform(-1.0, 1.0), 3)
            velocity_limit = round(random.uniform(0.1, 1.5), 3)
            safety_margin = round(random.uniform(0.01, 0.2), 3)
            rows.append(
                {
                    "trajectory_id": f"traj-{idx:04d}",
                    "ros_topic": ros_topic,
                    "actuator": actuator,
                    "target_position": target_position,
                    "velocity_limit": velocity_limit,
                    "safety_margin": safety_margin,
                }
            )

        if "csv" in spec.formats:
            with (dataset_root / "trajectories.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        if "json" in spec.formats:
            (dataset_root / "ros_topics.json").write_text(json.dumps(rows, indent=2))


__all__ = ["SyntheticDatasetBuilder"]
