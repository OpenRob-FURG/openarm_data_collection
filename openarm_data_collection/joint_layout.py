"""Layout canônico do vetor de 16 dims — fonte única para recorder, replay e
runner de policy.

Ordem (igual ao bi_openarm_follower do OpenArm LeRobot):
    [ left_joint_1..7, left_gripper, right_joint_1..7, right_gripper ]

Cada entrada mapeia o nome da feature LeRobot (`<side>_<joint>.pos`) para o nome
da junta no /joint_states e nos comandos dos controllers ROS.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
DEFAULT_GRIPPER_JOINT = {
    "left": "openarm_left_finger_joint1",
    "right": "openarm_right_finger_joint1",
}


@dataclass(frozen=True)
class Slot:
    index: int
    feature: str          # ex. "left_joint_3.pos"
    ros_joint: str         # ex. "openarm_left_joint3"
    side: str              # "left" | "right"
    is_gripper: bool


# Limites do URDF OpenArm **v1.0** (este robô). Juntas de braço em radianos;
# a garra (finger_joint1) é PRISMÁTICA -> metros, 0..0.044. (No v2.0 a garra é
# revoluta 0..0.785 — se trocar de robô, ajuste aqui.)
URDF_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "openarm_left_joint1": (-3.4907, 1.3963),
    "openarm_left_joint2": (-3.3161, 0.17453),
    "openarm_left_joint3": (-1.5708, 1.5708),
    "openarm_left_joint4": (-0.02, 2.4435),  # limite mecânico real do cotovelo (não hiperextende)
    "openarm_left_joint5": (-1.5708, 1.5708),
    "openarm_left_joint6": (-0.7854, 0.7854),
    "openarm_left_joint7": (-1.5708, 1.5708),
    "openarm_left_finger_joint1": (0.0, 0.044),
    "openarm_right_joint1": (-1.3963, 3.4907),
    "openarm_right_joint2": (-0.17453, 3.3161),
    "openarm_right_joint3": (-1.5708, 1.5708),
    "openarm_right_joint4": (-0.02, 2.4435),  # idem joint4 esquerdo
    "openarm_right_joint5": (-1.5708, 1.5708),
    "openarm_right_joint6": (-0.7854, 0.7854),
    "openarm_right_joint7": (-1.5708, 1.5708),
    "openarm_right_finger_joint1": (0.0, 0.044),
}


def build_slots(arm_joints=None, gripper_joint=None) -> list[Slot]:
    arm_joints = arm_joints or DEFAULT_ARM_JOINTS
    gripper_joint = gripper_joint or DEFAULT_GRIPPER_JOINT
    slots: list[Slot] = []
    k = 0
    for side in ("left", "right"):
        for n, ros_name in enumerate(arm_joints[side], start=1):
            slots.append(Slot(k, f"{side}_joint_{n}.pos", ros_name, side, False))
            k += 1
        slots.append(Slot(k, f"{side}_gripper.pos", gripper_joint[side], side, True))
        k += 1
    return slots


def feature_names(slots: list[Slot] | None = None) -> list[str]:
    return [s.feature for s in (slots or build_slots())]
