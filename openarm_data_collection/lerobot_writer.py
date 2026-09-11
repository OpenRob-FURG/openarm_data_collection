"""Wrapper fino sobre LeRobotDataset (codebase v3.0) para gravação de episódios.

Produz um dataset com o mesmo layout do pipeline CAN nativo do OpenArm LeRobot
(`bi_openarm_follower`): observation.state / action de 16 dims e câmeras em vídeo
AV1. Ver README.md para o schema completo.
"""
from __future__ import annotations

import os

# Dataset é local; nunca consultar o Hugging Face Hub (evita 401 em repo_id
# inexistente e trava offline). Definir antes de importar o lerobot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

CODEBASE_ROBOT_TYPE = "bi_openarm_follower"


def _looks_like_dataset(root_dir: str) -> bool:
    return os.path.isfile(os.path.join(root_dir, "meta", "info.json"))


class LeRobotWriter:
    def __init__(
        self,
        repo_id: str,
        root_dir: str,
        fps: int,
        features: dict,
        robot_type: str = CODEBASE_ROBOT_TYPE,
    ):
        self.repo_id = repo_id
        self.root_dir = root_dir
        self._episodes_saved = 0

        if _looks_like_dataset(root_dir):
            # Retoma um dataset local existente para continuar adicionando episódios.
            self.dataset = LeRobotDataset.resume(repo_id, root=root_dir)
            self._episodes_saved = self.dataset.meta.total_episodes
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=root_dir,
                robot_type=robot_type,
                use_videos=True,
            )

    @property
    def episodes_saved(self) -> int:
        return self._episodes_saved

    @property
    def current_episode_frames(self) -> int:
        writer = getattr(self.dataset, "writer", None)
        buf = getattr(writer, "episode_buffer", None) if writer else None
        return int(buf["size"]) if buf else 0

    def add_frame(self, frame: dict) -> None:
        """`frame` deve conter todas as features + 'task'. Arrays já vêm no dtype
        certo (float32 para estado/ação, uint8 HxWx3 para imagens)."""
        clean = {}
        for key, value in frame.items():
            if key == "task":
                clean[key] = value
            elif isinstance(value, np.ndarray):
                clean[key] = value
            else:
                clean[key] = np.asarray(value)
        self.dataset.add_frame(clean)

    def save_episode(self) -> int:
        self.dataset.save_episode()
        self._episodes_saved += 1
        return self._episodes_saved

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        # Idempotente: só finaliza se ainda houver escrita pendente.
        try:
            self.dataset.finalize()
        except Exception:
            pass
