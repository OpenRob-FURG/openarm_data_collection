"""Layout por braço (joint_layout.build_slots(arm=...)): python puro, sem ROS."""
import pytest

from openarm_data_collection.joint_layout import (
    ARMS,
    arm_tag,
    build_slots,
    camera_names_for_arm,
    feature_names,
    image_keys_for_arm,
    robot_type_for_arm,
    sides_for_arm,
)

# Golden: o layout bimanual gravado hoje. Se isto mudar, TODO dataset e checkpoint
# existente fica incompatível — por isso está escrito por extenso, não derivado.
GOLDEN_BOTH = [
    *(f"left_joint_{i}.pos" for i in range(1, 8)), "left_gripper.pos",
    *(f"right_joint_{i}.pos" for i in range(1, 8)), "right_gripper.pos",
]


def test_both_is_the_historic_layout():
    slots = build_slots()
    assert [s.feature for s in slots] == GOLDEN_BOTH
    assert [s.index for s in slots] == list(range(16))
    assert slots[0].ros_joint == "openarm_left_joint1"
    assert slots[7].ros_joint == "openarm_left_finger_joint1" and slots[7].is_gripper
    assert slots[15].ros_joint == "openarm_right_finger_joint1" and slots[15].is_gripper
    # default de `arm` tem que ser "both"
    assert build_slots(arm="both") == slots


@pytest.mark.parametrize("arm", ["left", "right"])
def test_single_arm_has_8_slots_indexed_from_zero(arm):
    slots = build_slots(arm=arm)
    assert len(slots) == 8
    assert [s.index for s in slots] == list(range(8))      # NÃO 8..15
    assert {s.side for s in slots} == {arm}
    assert [s.is_gripper for s in slots] == [False] * 7 + [True]


def test_single_arm_is_a_slice_of_both_in_the_same_order():
    # É o que mantém compatível um checkpoint treinado na fatia do dataset
    # bimanual (as stats do gdorneles/smolvla_openarm_right batem com dims 8..15).
    both = build_slots()
    for arm, sl in (("left", slice(0, 8)), ("right", slice(8, 16))):
        single = build_slots(arm=arm)
        assert [s.feature for s in single] == [s.feature for s in both[sl]]
        assert [s.ros_joint for s in single] == [s.ros_joint for s in both[sl]]


def test_feature_names_follow_arm():
    assert feature_names() == GOLDEN_BOTH
    assert feature_names(arm="right") == GOLDEN_BOTH[8:]
    assert feature_names(arm="left") == GOLDEN_BOTH[:8]
    # aceita slots já montados (o recorder passa os dele, com nomes de junta custom)
    assert feature_names(build_slots(arm="right")) == GOLDEN_BOTH[8:]


def test_custom_joint_names_are_respected_per_arm():
    arm_joints = {"left": [f"L{i}" for i in range(7)], "right": [f"R{i}" for i in range(7)]}
    grip = {"left": "LG", "right": "RG"}
    slots = build_slots(arm_joints, grip, arm="right")
    assert [s.ros_joint for s in slots] == [f"R{i}" for i in range(7)] + ["RG"]


def test_cameras_follow_the_arm_and_always_include_the_chest():
    assert camera_names_for_arm("both") == ["cam_chest", "left_cam_wrist", "right_cam_wrist"]
    assert camera_names_for_arm("right") == ["cam_chest", "right_cam_wrist"]
    assert camera_names_for_arm("left") == ["cam_chest", "left_cam_wrist"]
    assert image_keys_for_arm("right") == [
        "observation.images.cam_chest", "observation.images.right_cam_wrist"]


def test_tags_and_robot_types():
    assert arm_tag("both") == "bimanual"       # mantém ~/datasets/openarm_bimanual_vr
    assert arm_tag("right") == "right_arm"
    assert arm_tag("left") == "left_arm"
    assert robot_type_for_arm("both") == "bi_openarm_follower"
    assert robot_type_for_arm("right") == "openarm_follower"
    assert sides_for_arm("both") == ("left", "right")
    assert sides_for_arm("left") == ("left",)


@pytest.mark.parametrize("bad", ["", "esquerdo", "BOTH", "Right", None])
def test_invalid_arm_is_rejected_loudly(bad):
    with pytest.raises(ValueError):
        build_slots(arm=bad)
    with pytest.raises(ValueError):
        arm_tag(bad)


def test_arms_constant():
    assert ARMS == ("both", "left", "right")
