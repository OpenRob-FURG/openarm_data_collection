"""Valida um dataset gravado pelo episode_recorder.

    python -m openarm_data_collection.validate_dataset --root /path/do/dataset
    ros2 run openarm_data_collection validate_dataset --root /path

Checa: schema x layout esperado, contagem/duração dos episódios, consistência de
fps/timestamps e ranges de observation.state / action.

O braço do dataset (both = 16 dims e 3 câmeras | left/right = 8 dims, peito + pulso)
é detectado pelos nomes das dims; `--arm` só serve pra exigir um deles — se o dataset
for de outro braço, reprova em vez de aceitar em silêncio.
Opcional: --dump-episode N grava alguns frames de cada câmera como PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from openarm_data_collection.joint_layout import (
    ARMS, build_slots, feature_names, image_keys_for_arm,
)


def detect_arm(names) -> str | None:
    """both | left | right conforme os nomes das dims; None se não bate com nenhum."""
    names = list(names or [])
    for arm in ARMS:
        if names == feature_names(build_slots(arm=arm)):
            return arm
    return None


def _load(root: str):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        sys.exit(
            "lerobot não encontrado neste interpretador. Rode com o venv 3.12:\n"
            "  ~/openarm_ws/.venv312/bin/python -m "
            "openarm_data_collection.validate_dataset --root <path>")

    root_p = Path(root).expanduser()
    if not (root_p / "meta" / "info.json").is_file():
        sys.exit(f"não parece um dataset LeRobot: {root_p}")
    return LeRobotDataset(repo_id=f"local/{root_p.name}", root=str(root_p))


def _check_schema(meta, want_arm: str | None = None) -> tuple[list[str], str | None]:
    """(problemas, braço detectado). `want_arm` exige um braço específico."""
    problems = []
    feats = meta.features
    arm = detect_arm((feats.get("observation.state") or {}).get("names"))
    if arm is None:
        problems.append(
            "observation.state: os nomes das dims não batem com nenhum layout "
            f"({' | '.join(ARMS)})\n  got={(feats.get('observation.state') or {}).get('names')}")
    elif want_arm and arm != want_arm:
        problems.append(f"o dataset é arm={arm}, mas --arm={want_arm} foi exigido")
    if arm is None:
        return problems, None

    names = feature_names(build_slots(arm=arm))
    for key in ("observation.state", "action"):
        if key not in feats:
            problems.append(f"falta feature {key}")
            continue
        shape = tuple(feats[key]["shape"])
        if shape != (len(names),):
            problems.append(f"{key}: shape {shape}, esperado ({len(names)},)")
        got = feats[key].get("names")
        if got and list(got) != names:
            problems.append(f"{key}: names diferem do layout arm={arm}\n  got={got}")
    expected_imgs = set(image_keys_for_arm(arm))
    missing = expected_imgs - set(feats)
    if missing:
        problems.append(f"faltam câmeras: {sorted(missing)}")
    extras = {k for k in feats if k.startswith("observation.images.")} - expected_imgs
    if extras:
        problems.append(f"câmeras que arm={arm} não deveria ter: {sorted(extras)}")
    for key in expected_imgs & set(feats):
        if feats[key]["dtype"] != "video":
            problems.append(f"{key}: dtype {feats[key]['dtype']}, esperado 'video'")
    if getattr(meta, "fps", None) != 30:
        problems.append(f"fps = {getattr(meta, 'fps', None)} (referência usa 30)")
    return problems, arm


def _episode_stats(ds) -> list[str]:
    lines = []
    hf = ds.hf_dataset
    ep_col = np.asarray(hf["episode_index"])
    ts_col = np.asarray(hf["timestamp"], dtype=np.float64)
    fps = ds.meta.fps
    for ep in sorted(set(int(x) for x in ep_col)):
        m = ep_col == ep
        ts = ts_col[m]
        n = int(m.sum())
        dur = float(ts[-1] - ts[0]) if n > 1 else 0.0
        dt = np.diff(ts)
        jitter = float(np.max(np.abs(dt - 1.0 / fps))) if len(dt) else 0.0
        flag = "  ⚠️ jitter" if jitter > 0.5 / fps else ""
        lines.append(f"  ep {ep:>3}: {n:>4} frames  {dur:6.2f}s  "
                     f"~{n / dur if dur else 0:4.1f} fps  jitter={jitter*1000:4.1f}ms{flag}")
    return lines


def _range_report(ds, names: list[str]) -> list[str]:
    hf = ds.hf_dataset
    out = []
    for key in ("observation.state", "action"):
        arr = np.asarray(hf[key], dtype=np.float32)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        out.append(f"  {key}:")
        for i, name in enumerate(names):
            out.append(f"    {name:<18} [{lo[i]:8.3f}, {hi[i]:8.3f}]")
    return out


def _dump_episode(ds, ep: int, dst: str):
    import PIL.Image

    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    hf = ds.hf_dataset
    ep_col = np.asarray(hf["episode_index"])
    idxs = np.where(ep_col == ep)[0]
    picks = idxs[:: max(1, len(idxs) // 5)][:5]
    for gi in picks:
        item = ds[int(gi)]
        for key in sorted(k for k in ds.meta.features if k.startswith("observation.images.")):
            img = item[key]
            arr = (img.numpy() if hasattr(img, "numpy") else np.asarray(img))
            if arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            name = key.split(".")[-1]
            PIL.Image.fromarray(arr).save(out / f"ep{ep}_f{int(gi)}_{name}.png")
    print(f"frames salvos em {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", choices=ARMS, default=None,
                    help="exige este braço (padrão: detecta pelo schema e aceita qualquer um)")
    ap.add_argument("--dump-episode", type=int, default=None)
    ap.add_argument("--dump-dir", default="/tmp/dataset_preview")
    args, _ = ap.parse_known_args(argv)

    ds = _load(args.root)
    meta = ds.meta
    import json
    info = json.loads((Path(args.root).expanduser() / "meta" / "info.json").read_text())
    print(f"\n=== {args.root} ===")
    print(f"codebase={info.get('codebase_version', '?')}  "
          f"robot_type={info.get('robot_type', '?')}  fps={getattr(meta, 'fps', '?')}")
    print(f"episódios={meta.total_episodes}  frames={meta.total_frames}  tarefas={meta.total_tasks}")

    problems, arm = _check_schema(meta, args.arm)
    print("\n-- schema --")
    print(f"  OK (arm={arm}: {len(feature_names(build_slots(arm=arm)))} dims, "
          f"câmeras {[k.split('.')[-1] for k in image_keys_for_arm(arm)]})" if not problems
          else "\n".join(f"  ❌ {p}" for p in problems))

    print("\n-- episódios --")
    print("\n".join(_episode_stats(ds)))
    if arm is not None:
        print("\n-- ranges --")
        print("\n".join(_range_report(ds, feature_names(build_slots(arm=arm)))))

    if args.dump_episode is not None:
        print()
        _dump_episode(ds, args.dump_episode, args.dump_dir)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
