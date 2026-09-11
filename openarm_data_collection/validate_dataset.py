"""Valida um dataset gravado pelo episode_recorder.

    python -m openarm_data_collection.validate_dataset --root /path/do/dataset
    ros2 run openarm_data_collection validate_dataset --root /path

Checa: schema x formato bi_openarm_follower esperado, contagem/duração dos
episódios, consistência de fps/timestamps e ranges de observation.state / action.
Opcional: --dump-episode N grava alguns frames de cada câmera como PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXPECTED_STATE_NAMES = [
    *(f"left_joint_{i}.pos" for i in range(1, 8)), "left_gripper.pos",
    *(f"right_joint_{i}.pos" for i in range(1, 8)), "right_gripper.pos",
]
EXPECTED_IMAGE_KEYS = {
    "observation.images.cam_chest",
    "observation.images.left_cam_wrist",
    "observation.images.right_cam_wrist",
}


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


def _check_schema(meta) -> list[str]:
    problems = []
    feats = meta.features
    for key in ("observation.state", "action"):
        if key not in feats:
            problems.append(f"falta feature {key}")
            continue
        shape = tuple(feats[key]["shape"])
        if shape != (16,):
            problems.append(f"{key}: shape {shape}, esperado (16,)")
        names = feats[key].get("names")
        if names and list(names) != EXPECTED_STATE_NAMES:
            problems.append(f"{key}: names diferem do bi_openarm_follower\n  got={names}")
    missing = EXPECTED_IMAGE_KEYS - set(feats)
    if missing:
        problems.append(f"faltam câmeras: {sorted(missing)}")
    for key in EXPECTED_IMAGE_KEYS & set(feats):
        if feats[key]["dtype"] != "video":
            problems.append(f"{key}: dtype {feats[key]['dtype']}, esperado 'video'")
    if getattr(meta, "fps", None) != 30:
        problems.append(f"fps = {getattr(meta, 'fps', None)} (referência usa 30)")
    return problems


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


def _range_report(ds) -> list[str]:
    hf = ds.hf_dataset
    out = []
    for key in ("observation.state", "action"):
        arr = np.asarray(hf[key], dtype=np.float32)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        out.append(f"  {key}:")
        for i, name in enumerate(EXPECTED_STATE_NAMES):
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
        for key in EXPECTED_IMAGE_KEYS & set(ds.meta.features):
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

    problems = _check_schema(meta)
    print("\n-- schema --")
    print("  OK (compatível com bi_openarm_follower)" if not problems
          else "\n".join(f"  ❌ {p}" for p in problems))

    print("\n-- episódios --")
    print("\n".join(_episode_stats(ds)))
    print("\n-- ranges --")
    print("\n".join(_range_report(ds)))

    if args.dump_episode is not None:
        print()
        _dump_episode(ds, args.dump_episode, args.dump_dir)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
