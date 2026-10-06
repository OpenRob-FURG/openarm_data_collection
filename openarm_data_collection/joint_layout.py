"""Layout canônico do vetor de estado/ação — fonte única para recorder, replay,
validate e runner de policy.

Ordem com `arm="both"` (16 dims, igual ao bi_openarm_follower do OpenArm LeRobot):
    [ left_joint_1..7, left_gripper, right_joint_1..7, right_gripper ]
Com `arm="left"` ou `"right"`: só os 8 do respectivo braço, mesma ordem e mesmos
nomes (`right_joint_1.pos` …), reindexados de 0 a 7.

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


# Qual(is) braço(s) um dataset / uma policy / o pipeline cobre. `both` é o
# padrão e o layout de sempre (16 dims); `left`/`right` são 8 dims.
ARMS = ("both", "left", "right")

_ARM_TAG = {"both": "bimanual", "left": "left_arm", "right": "right_arm"}


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm={arm!r} inválido (use {' | '.join(ARMS)})")
    return arm


def sides_for_arm(arm: str) -> tuple[str, ...]:
    return ("left", "right") if check_arm(arm) == "both" else (arm,)


def arm_tag(arm: str) -> str:
    """Sufixo de nome de dataset: both->bimanual, right->right_arm, left->left_arm."""
    return _ARM_TAG[check_arm(arm)]


def robot_type_for_arm(arm: str) -> str:
    return "bi_openarm_follower" if check_arm(arm) == "both" else "openarm_follower"


def camera_names_for_arm(arm: str) -> list[str]:
    """Câmeras que um dataset desse braço grava: o peito + o pulso de cada braço
    coberto. A câmera do pulso do braço ocioso não entra — ela só mostraria um
    braço parado, e o synchronizer do recorder só dispara com TODAS as assinadas."""
    return ["cam_chest", *(f"{side}_cam_wrist" for side in sides_for_arm(arm))]


def image_keys_for_arm(arm: str) -> list[str]:
    return [f"observation.images.{n}" for n in camera_names_for_arm(arm)]


def build_slots(arm_joints=None, gripper_joint=None, arm: str = "both") -> list[Slot]:
    """Slots do vetor de estado/ação, na ordem canônica.

    `index` é a posição NO VETOR DESSE `arm`: com `arm="right"` os 8 slots vão de
    0 a 7 (não de 8 a 15). Recorder, runner, validate e replay chamam esta mesma
    função com o mesmo `arm`, então a ordem não tem como divergir entre gravar e
    executar.
    """
    arm_joints = arm_joints or DEFAULT_ARM_JOINTS
    gripper_joint = gripper_joint or DEFAULT_GRIPPER_JOINT
    slots: list[Slot] = []
    k = 0
    for side in sides_for_arm(arm):
        for n, ros_name in enumerate(arm_joints[side], start=1):
            slots.append(Slot(k, f"{side}_joint_{n}.pos", ros_name, side, False))
            k += 1
        slots.append(Slot(k, f"{side}_gripper.pos", gripper_joint[side], side, True))
        k += 1
    return slots


def feature_names(slots: list[Slot] | None = None, arm: str = "both") -> list[str]:
    return [s.feature for s in (slots or build_slots(arm=arm))]
