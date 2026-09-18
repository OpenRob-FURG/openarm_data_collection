"""episode_recorder — grava episódios de teleop bimanual no formato LeRobot v3.

Fontes ROS:
  * /joint_states  (sensor_msgs/JointState)  -> observation.state (16 dims)
  * braco: comando do forward_position_controller (std_msgs/Float64MultiArray)
    garra: controller_state do gripper_controller (control_msgs)  -> action (16 dims)
  * 3 câmeras (sensor_msgs/Image ou CompressedImage) -> observation.images.*

Layout do vetor de 16 dims (igual ao bi_openarm_follower do OpenArm LeRobot):
  [ left_joint_1..7, left_gripper, right_joint_1..7, right_gripper ]

Controle por botão (TeleopXR / Meta Quest), via xr_buttons.XRButtonMonitor:
  A (tap)          -> START   começa um episódio
  B (tap)          -> STOP    salva o episódio e dispara ida pra home
  X (long-press)   -> DISCARD descarta o episódio em andamento
"""
from __future__ import annotations

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Empty, Float64MultiArray

from openarm_data_collection.image_utils import to_dataset_frame
from openarm_data_collection.lerobot_writer import LeRobotWriter
from openarm_data_collection.joint_layout import build_slots, feature_names
from openarm_data_collection.xr_buttons import RecordingEvent, XRButtonMonitor


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("episode_recorder")

        # ---- parâmetros -------------------------------------------------
        p = self.declare_parameter
        self.task_name = p("task_name", "pegar o copo vermelho").value
        self.dataset_root = p("dataset_root", "/home/popzap-lab/datasets/openarm_bimanual_vr").value
        self.repo_id = p("repo_id", "furg/openarm_bimanual_vr").value
        self.fps = int(p("fps", 30).value)

        self.img_h = int(p("image_height", 480).value)
        self.img_w = int(p("image_width", 640).value)
        self.min_episode_frames = int(p("min_episode_frames", 15).value)
        self.sync_slop = float(p("sync_slop", 0.033).value)

        # radianos -> unidade gravada. 1.0 mantém radianos; 57.29578 grava em graus.
        self.arm_angle_scale = float(p("arm_angle_scale", 1.0).value)
        self.gripper_scale = float(p("gripper_scale", 1.0).value)
        self.gripper_offset = float(p("gripper_offset", 0.0).value)

        # nomes das juntas no /joint_states e nos comandos
        arm_joints = {
            "left": list(p("left_arm_joint_names",
                           [f"openarm_left_joint{i}" for i in range(1, 8)]).value),
            "right": list(p("right_arm_joint_names",
                            [f"openarm_right_joint{i}" for i in range(1, 8)]).value),
        }
        gripper_joint = {
            "left": p("left_gripper_joint_name", "openarm_left_finger_joint1").value,
            "right": p("right_gripper_joint_name", "openarm_right_finger_joint1").value,
        }
        self.slots = build_slots(arm_joints, gripper_joint)

        # tópicos
        cam_chest = p("cam_chest_topic", "/cam_chest/color/image_raw").value
        cam_lw = p("left_wrist_topic", "/left_wrist/image_raw").value
        cam_rw = p("right_wrist_topic", "/right_wrist/image_raw").value
        cam_compressed = bool(p("cameras_compressed", False).value)

        # O teleop comanda o braço pelo forward_position_controller, que publica
        # Float64MultiArray (sem nomes de junta) — daí o par tópico/nomes abaixo.
        # Pôr aqui um tópico de JointTrajectory faria o recorder não ver comando
        # nenhum e o `action` cair no fallback, virando cópia do `state`.
        left_arm_cmd = p("left_arm_cmd_topic",
                         "/left_forward_position_controller/commands").value
        right_arm_cmd = p("right_arm_cmd_topic",
                          "/right_forward_position_controller/commands").value
        # Garra: controller_state, não o tópico de comando. O teleop manda a
        # garra por *action*, o home_mover por tópico — e o `reference` é o
        # setpoint que o controller segura em qualquer um dos casos.
        left_grip_cmd = p("left_gripper_cmd_topic",
                          "/left_gripper_controller/controller_state").value
        right_grip_cmd = p("right_gripper_cmd_topic",
                           "/right_gripper_controller/controller_state").value

        # ---- dataset --------------------------------------------------
        names = feature_names(self.slots)
        features = {
            "observation.state": {"dtype": "float32", "shape": (16,), "names": names},
            "action": {"dtype": "float32", "shape": (16,), "names": names},
            "observation.images.cam_chest": {
                "dtype": "video", "shape": (self.img_h, self.img_w, 3),
                "names": ["height", "width", "channels"]},
            "observation.images.left_cam_wrist": {
                "dtype": "video", "shape": (self.img_h, self.img_w, 3),
                "names": ["height", "width", "channels"]},
            "observation.images.right_cam_wrist": {
                "dtype": "video", "shape": (self.img_h, self.img_w, 3),
                "names": ["height", "width", "channels"]},
        }
        self.writer = LeRobotWriter(
            repo_id=self.repo_id, root_dir=self.dataset_root,
            fps=self.fps, features=features,
        )

        # ---- estado interno -----------------------------------------
        self.is_recording = False
        # ação = alvo comandado; mantém o último valor por junta ROS.
        self._cmd: dict[str, float] = {}
        self._frames_this_episode = 0

        # ---- publishers / subs -------------------------------------
        self.home_trigger_pub = self.create_publisher(Empty, "/data_collection/go_home", 1)

        # BEST_EFFORT: o bimanual_teleop publica o comando do braço a 100 Hz em
        # BEST_EFFORT. Um sub RELIABLE é incompatível com esse publisher e o
        # recorder não recebia NADA dele ("offering incompatible QoS") — a
        # coluna de ação do dataset ficava só com o que o home_mover mandava.
        # BEST_EFFORT aceita os dois (um pub RELIABLE serve sub BEST_EFFORT).
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for topic, joint_names in ((left_arm_cmd, arm_joints["left"]),
                                   (right_arm_cmd, arm_joints["right"])):
            self.create_subscription(
                Float64MultiArray, topic,
                lambda msg, n=joint_names: self._on_array_command(msg, n), cmd_qos)

        for topic in (left_grip_cmd, right_grip_cmd):
            self.create_subscription(
                JointTrajectoryControllerState, topic, self._on_controller_state, 10)

        img_type = CompressedImage if cam_compressed else Image
        sub_js = Subscriber(self, JointState, "/joint_states")
        sub_chest = Subscriber(self, img_type, cam_chest)
        sub_lw = Subscriber(self, img_type, cam_lw)
        sub_rw = Subscriber(self, img_type, cam_rw)
        # slop de meio frame alinha /joint_states (>=50 Hz) com as câmeras a `fps`.
        self.sync = ApproximateTimeSynchronizer(
            [sub_js, sub_chest, sub_lw, sub_rw],
            queue_size=30, slop=self.sync_slop)
        self.sync.registerCallback(self._on_synced)

        self.buttons = XRButtonMonitor(self, self._on_button_event)

        self.get_logger().info(
            f"episode_recorder pronto | tarefa='{self.task_name}' | "
            f"dataset={self.dataset_root} | episódios já gravados="
            f"{self.writer.episodes_saved}. A=start B=stop X(long)=discard.")

    # ------------------------------------------------------------------
    def _on_array_command(self, msg: Float64MultiArray, names: list[str]):
        # Float64MultiArray não carrega nomes: a ordem do array é a mesma do
        # `joints:` do forward_position_controller, ou seja joint1..joint7.
        for name, value in zip(names, msg.data):
            self._cmd[name] = float(value)

    def _on_controller_state(self, msg: JointTrajectoryControllerState):
        for name, value in zip(msg.joint_names, msg.reference.positions):
            self._cmd[name] = float(value)

    def _on_button_event(self, event: RecordingEvent):
        if event is RecordingEvent.START:
            if self.is_recording:
                return
            self.is_recording = True
            self._frames_this_episode = 0
            self.get_logger().info(f"▶️  REC episódio #{self.writer.episodes_saved + 1}")

        elif event is RecordingEvent.HOME:  # B = parar + salvar
            if not self.is_recording:
                return
            self.is_recording = False
            n = self._frames_this_episode
            if n < self.min_episode_frames:
                self.writer.discard_episode()
                self.get_logger().warn(
                    f"episódio curto demais ({n} < {self.min_episode_frames}) — descartado")
            else:
                total = self.writer.save_episode()
                self.get_logger().info(f"💾  episódio salvo ({n} frames) | total={total}")
            self.home_trigger_pub.publish(Empty())

        elif event is RecordingEvent.DISCARD:
            if not self.is_recording:
                return
            self.is_recording = False
            self.writer.discard_episode()
            self.get_logger().warn("🗑️  episódio descartado")

    # ------------------------------------------------------------------
    def _scale(self, slot, raw: float) -> float:
        if slot.is_gripper:
            return raw * self.gripper_scale + self.gripper_offset
        return raw * self.arm_angle_scale

    def _state_vector(self, js: JointState) -> np.ndarray:
        idx = {n: i for i, n in enumerate(js.name)}
        out = np.zeros(16, dtype=np.float32)
        for s in self.slots:
            if s.ros_joint in idx:
                out[s.index] = self._scale(s, js.position[idx[s.ros_joint]])
        return out

    def _action_vector(self, state: np.ndarray) -> np.ndarray:
        out = state.copy()  # fallback: junta ainda não comandada segue o estado
        for s in self.slots:
            if s.ros_joint in self._cmd:
                out[s.index] = self._scale(s, self._cmd[s.ros_joint])
        return out.astype(np.float32)

    def _on_synced(self, js: JointState, chest, left_wrist, right_wrist):
        if not self.is_recording:
            return
        try:
            state = self._state_vector(js)
            frame = {
                "observation.state": state,
                "action": self._action_vector(state),
                "observation.images.cam_chest": to_dataset_frame(chest, self.img_h, self.img_w),
                "observation.images.left_cam_wrist": to_dataset_frame(left_wrist, self.img_h, self.img_w),
                "observation.images.right_cam_wrist": to_dataset_frame(right_wrist, self.img_h, self.img_w),
                "task": self.task_name,
            }
            self.writer.add_frame(frame)
            self._frames_this_episode += 1
        except Exception as exc:  # nunca deixa o callback derrubar o nó no meio da coleta
            self.get_logger().error(f"frame ignorado: {exc}")

    def shutdown(self):
        if self.is_recording and self._frames_this_episode >= self.min_episode_frames:
            self.get_logger().info("shutdown durante gravação — salvando episódio parcial")
            self.writer.save_episode()
        self.writer.finalize()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
