"""episode_recorder — grava episódios de teleop no formato LeRobot v3.

`arm` (both | left | right, padrão both) escolhe o que entra no dataset:

  both   16 dims, 3 câmeras  — o bimanual de sempre (bi_openarm_follower)
  right  8 dims, cam_chest + right_cam_wrist  (openarm_follower)
  left   8 dims, cam_chest + left_cam_wrist   (openarm_follower)

Fontes ROS (só as do braço escolhido):
  * /joint_states  (sensor_msgs/JointState)  -> observation.state
  * braco: comando do forward_position_controller (std_msgs/Float64MultiArray)
    garra: controller_state do gripper_controller (control_msgs)  -> action
  * câmeras (sensor_msgs/Image ou CompressedImage) -> observation.images.*

Layout do vetor (joint_layout.build_slots — fonte única, a mesma do runner):
  both:  [ left_joint_1..7, left_gripper, right_joint_1..7, right_gripper ]
  right: [ right_joint_1..7, right_gripper ]   (mesmos nomes, reindexado de 0)

Controle por botão (TeleopXR / Meta Quest), via xr_buttons.XRButtonMonitor:
  A (tap)          -> START   começa um episódio
  B (tap)          -> STOP    salva o episódio e dispara ida pra home
  X (long-press)   -> DISCARD descarta o episódio em andamento
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Empty, Float64MultiArray

from openarm_data_collection.image_utils import to_dataset_frame
from openarm_data_collection.lerobot_writer import LeRobotWriter
from openarm_data_collection.joint_layout import (
    arm_tag, build_slots, camera_names_for_arm, check_arm, feature_names,
    robot_type_for_arm, sides_for_arm,
)
from openarm_data_collection.xr_buttons import RecordingEvent, XRButtonMonitor


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("episode_recorder")

        # ---- parâmetros -------------------------------------------------
        p = self.declare_parameter
        self.task_name = p("task_name", "pegar o copo vermelho").value
        # O parser de parametros do ROS 2 nao expande nem `~` nem `$VAR` --
        # a string chega literal. Expandimos aqui para o yaml poder trazer um
        # caminho portatil ("~/datasets/...") em vez do home de uma maquina so,
        # que virava conflito toda vez que o repo ia pro outro PC. O expandvars
        # fica junto de graca, para apontar um disco externo via $VAR se precisar.
        #
        # `arm` (both|left|right) decide o schema do dataset: both = 16 dims e 3
        # câmeras (o de sempre); left/right = 8 dims, peito + pulso daquele braço.
        # `{arm_tag}` em dataset_root/repo_id vira bimanual|left_arm|right_arm, então
        # cada tipo de dataset ganha o próprio diretório e `both` continua em
        # ~/datasets/openarm_bimanual_vr. Um root explícito sem a marca vale como veio.
        self.arm = check_arm(p("arm", "both").value)
        tag = arm_tag(self.arm)
        self.dataset_root = os.path.expanduser(os.path.expandvars(
            p("dataset_root", "~/datasets/openarm_{arm_tag}_vr").value)).replace("{arm_tag}", tag)
        self.repo_id = p("repo_id", "furg/openarm_{arm_tag}_vr").value.replace("{arm_tag}", tag)
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
        self.slots = build_slots(arm_joints, gripper_joint, arm=self.arm)
        self.n_dims = len(self.slots)
        sides = sides_for_arm(self.arm)

        # tópicos
        cam_chest = p("cam_chest_topic", "/cam_chest/color/image_raw").value
        cam_lw = p("left_wrist_topic", "/left_wrist/image_raw").value
        cam_rw = p("right_wrist_topic", "/right_wrist/image_raw").value
        cam_compressed = bool(p("cameras_compressed", False).value)
        # nome da câmera no dataset -> tópico; só as de `camera_names_for_arm` entram
        cam_topics = {"cam_chest": cam_chest, "left_cam_wrist": cam_lw,
                      "right_cam_wrist": cam_rw}
        self.cam_names = camera_names_for_arm(self.arm)

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
            "observation.state": {"dtype": "float32", "shape": (self.n_dims,), "names": names},
            "action": {"dtype": "float32", "shape": (self.n_dims,), "names": names},
        }
        for cam in self.cam_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video", "shape": (self.img_h, self.img_w, 3),
                "names": ["height", "width", "channels"]}
        self.writer = LeRobotWriter(
            repo_id=self.repo_id, root_dir=self.dataset_root,
            fps=self.fps, features=features, robot_type=robot_type_for_arm(self.arm),
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
        arm_cmd = {"left": left_arm_cmd, "right": right_arm_cmd}
        grip_cmd = {"left": left_grip_cmd, "right": right_grip_cmd}
        for side in sides:   # o braço fora de --arm não entra no dataset
            self.create_subscription(
                Float64MultiArray, arm_cmd[side],
                lambda msg, n=arm_joints[side]: self._on_array_command(msg, n), cmd_qos)
            self.create_subscription(
                JointTrajectoryControllerState, grip_cmd[side],
                self._on_controller_state, 10)

        img_type = CompressedImage if cam_compressed else Image
        # Só as câmeras do braço: o synchronizer só dispara quando TODOS os tópicos
        # assinados têm mensagem, então assinar a câmera do pulso ocioso faria a
        # gravação inteira parar se ela caísse (e as C270 caem — ver recording_params).
        subs = [Subscriber(self, JointState, "/joint_states")]
        subs += [Subscriber(self, img_type, cam_topics[c]) for c in self.cam_names]
        # slop de meio frame alinha /joint_states (>=50 Hz) com as câmeras a `fps`.
        self.sync = ApproximateTimeSynchronizer(
            subs, queue_size=30, slop=self.sync_slop)
        self.sync.registerCallback(self._on_synced)

        self.buttons = XRButtonMonitor(self, self._on_button_event)

        self.get_logger().info(
            f"episode_recorder pronto | arm={self.arm} ({self.n_dims} dims, "
            f"câmeras={self.cam_names}) | tarefa='{self.task_name}' | "
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
                    f"episódio #{self.writer.episodes_saved + 1} curto demais "
                    f"({n} < {self.min_episode_frames}) — descartado")
            else:
                total = self.writer.save_episode()
                self.get_logger().info(f"💾  episódio salvo ({n} frames) | total={total}")
            self.home_trigger_pub.publish(Empty())

        elif event is RecordingEvent.DISCARD:
            if not self.is_recording:
                return
            self.is_recording = False
            self.writer.discard_episode()
            self.get_logger().warn(
                f"🗑️  episódio #{self.writer.episodes_saved + 1} descartado")

    # ------------------------------------------------------------------
    def _scale(self, slot, raw: float) -> float:
        if slot.is_gripper:
            return raw * self.gripper_scale + self.gripper_offset
        return raw * self.arm_angle_scale

    def _state_vector(self, js: JointState) -> np.ndarray:
        idx = {n: i for i, n in enumerate(js.name)}
        out = np.zeros(self.n_dims, dtype=np.float32)
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

    def _on_synced(self, js: JointState, *imgs):
        if not self.is_recording:
            return
        try:
            state = self._state_vector(js)
            frame = {
                "observation.state": state,
                "action": self._action_vector(state),
                "task": self.task_name,
            }
            for cam, msg in zip(self.cam_names, imgs):
                frame[f"observation.images.{cam}"] = to_dataset_frame(msg, self.img_h, self.img_w)
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
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGINT ou SIGTERM
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
