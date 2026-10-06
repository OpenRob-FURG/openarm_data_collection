"""dataset_replay — republica o `action` de um episódio gravado nos controllers.

Prova, SEM modelo, que a unidade (radiano), a ordem das juntas e a fiação dos
controllers estão corretas: se o braço reproduz o movimento gravado, o pipeline
de dados está consistente.

Funciona com dataset de qualquer `arm` (both = 16 dims, left/right = 8): as colunas
são mapeadas pelo NOME da feature, então só os controllers das juntas que existem
no dataset recebem comando — o braço que ficou de fora não é tocado.

    # 1) checagem seca — não move nada, só compara os ranges com os limites do URDF
    ros2 run openarm_data_collection dataset_replay \\
        --root /home/popzap-lab/datasets/openarm_bimanual_vr --episode 0 --dry-run

    # 2) replay lento (25% da velocidade), começando por ir devagar até o 1º frame
    ros2 run openarm_data_collection dataset_replay \\
        --root <path> --episode 0 --rate-scale 0.25 --goto-start

Precisa do venv 3.12 (lê o dataset via lerobot). Rode com:
    ~/openarm_ws/.venv312/bin/python -m openarm_data_collection.dataset_replay ...
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_data_collection.joint_layout import URDF_LIMITS_RAD, build_slots
from openarm_data_collection.validate_dataset import detect_arm


def _load_actions(root: str, episode: int):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        sys.exit("lerobot não encontrado — use ~/openarm_ws/.venv312/bin/python -m "
                 "openarm_data_collection.dataset_replay ...")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from pathlib import Path
    ds = LeRobotDataset(repo_id=f"local/{Path(root).name}", root=root)
    hf = ds.hf_dataset
    ep = np.asarray(hf["episode_index"])
    mask = np.where(ep == episode)[0]
    if len(mask) == 0:
        sys.exit(f"episódio {episode} não existe (episódios: {sorted(set(int(x) for x in ep))})")
    actions = np.asarray(hf.select(mask.tolist())["action"], dtype=np.float64)
    names = ds.meta.features["action"]["names"]
    return actions, names, ds.meta.fps


class Replayer(Node):
    def __init__(self, args):
        super().__init__("dataset_replay")
        self.slots = build_slots()
        self.arm_scale = args.arm_angle_scale
        self.grip_scale = args.gripper_scale
        self.grip_offset = args.gripper_offset

        self.pub = {
            ("left", False): self.create_publisher(
                JointTrajectory, args.left_arm_topic, 10),
            ("right", False): self.create_publisher(
                JointTrajectory, args.right_arm_topic, 10),
            ("left", True): self.create_publisher(
                JointTrajectory, args.left_gripper_topic, 10),
            ("right", True): self.create_publisher(
                JointTrajectory, args.right_gripper_topic, 10),
        }
        self._js = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)

    def _on_js(self, msg):
        self._js = {n: p for n, p in zip(msg.name, msg.position)}

    # dataset -> radianos ROS
    def _to_ros(self, slot, val: float) -> float:
        if slot.is_gripper:
            return (val - self.grip_offset) / self.grip_scale
        return val / self.arm_scale

    def rows_to_ros(self, actions: np.ndarray, names: list[str]) -> dict:
        col = {n: i for i, n in enumerate(names)}
        out = {s.ros_joint: np.array([self._to_ros(s, a[col[s.feature]]) for a in actions])
               for s in self.slots if s.feature in col}
        return out

    def check_limits(self, per_joint: dict) -> bool:
        ok = True
        self.get_logger().info("ranges do episódio (rad) vs limites do URDF:")
        for jn, series in per_joint.items():
            lo, hi = float(series.min()), float(series.max())
            ulo, uhi = URDF_LIMITS_RAD.get(jn, (-np.inf, np.inf))
            bad = lo < ulo - 1e-3 or hi > uhi + 1e-3
            ok &= not bad
            flag = "  ❌ FORA DO LIMITE" if bad else ""
            self.get_logger().info(
                f"  {jn:<26} [{lo:+.3f}, {hi:+.3f}]  urdf[{ulo:+.3f}, {uhi:+.3f}]{flag}")
        return ok

    def publish_frame(self, per_joint: dict, i: int, dt: float):
        for (side, is_grip), pub in self.pub.items():
            joints = [s.ros_joint for s in self.slots
                      if s.side == side and s.is_gripper == is_grip and s.ros_joint in per_joint]
            if not joints:
                continue
            m = JointTrajectory()
            m.header.stamp = self.get_clock().now().to_msg()
            m.joint_names = joints
            pt = JointTrajectoryPoint()
            pt.positions = [float(per_joint[j][i]) for j in joints]
            pt.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
            m.points = [pt]
            pub.publish(m)

    def goto_start(self, per_joint: dict, seconds: float):
        if self._js is None:
            self.get_logger().warn("sem /joint_states — pulando goto-start")
            return
        self.get_logger().info(f"indo devagar até o 1º frame em {seconds}s...")
        for (side, is_grip), pub in self.pub.items():
            joints = [s.ros_joint for s in self.slots
                      if s.side == side and s.is_gripper == is_grip and s.ros_joint in per_joint]
            if not joints:
                continue
            m = JointTrajectory()
            m.header.stamp = self.get_clock().now().to_msg()
            m.joint_names = joints
            pt = JointTrajectoryPoint()
            pt.positions = [float(per_joint[j][0]) for j in joints]
            pt.time_from_start = Duration(sec=int(seconds), nanosec=0)
            m.points = [pt]
            pub.publish(m)
        time.sleep(seconds + 0.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate-scale", type=float, default=1.0,
                    help="fração da velocidade original (0.25 = 4x mais lento)")
    ap.add_argument("--skip-limit-check", action="store_true",
                    help="não aborta se o episódio exceder os limites do URDF (avisa e segue)")
    ap.add_argument("--goto-start", action="store_true",
                    help="move devagar até o 1º frame antes de começar")
    ap.add_argument("--goto-start-seconds", type=float, default=5.0)
    ap.add_argument("--arm-angle-scale", type=float, default=1.0)
    ap.add_argument("--gripper-scale", type=float, default=1.0)
    ap.add_argument("--gripper-offset", type=float, default=0.0)
    ap.add_argument("--left-arm-topic", default="/left_joint_trajectory_controller/joint_trajectory")
    ap.add_argument("--right-arm-topic", default="/right_joint_trajectory_controller/joint_trajectory")
    ap.add_argument("--left-gripper-topic", default="/left_gripper_controller/joint_trajectory")
    ap.add_argument("--right-gripper-topic", default="/right_gripper_controller/joint_trajectory")
    args, _ = ap.parse_known_args(argv)

    actions, names, fps = _load_actions(args.root, args.episode)

    rclpy.init()
    node = Replayer(args)
    per_joint = node.rows_to_ros(actions, names)

    node.get_logger().info(
        f"episódio {args.episode}: {len(actions)} frames @ {fps} fps | "
        f"dataset arm={detect_arm(names)} ({len(names)} dims)")
    within = node.check_limits(per_joint)

    if args.dry_run:
        node.get_logger().info("DRY-RUN: nada publicado. " +
                               ("limites OK ✅" if within else "⚠️ há juntas fora do limite"))
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if within else 1)

    if not within and not args.skip_limit_check:
        node.get_logger().error("abortando: episódio excede limites do URDF "
                                "(--dry-run pra ver, --skip-limit-check pra forçar)")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    if not within:
        node.get_logger().warn("⚠️ episódio fora dos limites do URDF — seguindo (--skip-limit-check)")

    # deixa o /joint_states chegar
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.05)
        if node._js is not None:
            break

    if args.goto_start:
        node.goto_start(per_joint, args.goto_start_seconds)

    dt = 1.0 / (fps * max(args.rate_scale, 1e-3))
    node.get_logger().info(f"replay: {len(actions)} frames, dt={dt*1000:.1f}ms "
                           f"({args.rate_scale:.2f}x). Ctrl-C pra abortar.")
    try:
        for i in range(len(actions)):
            node.publish_frame(per_joint, i, dt)
            end = time.time() + dt
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.005)
    except KeyboardInterrupt:
        node.get_logger().warn("replay abortado")
    node.get_logger().info("replay fim")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
