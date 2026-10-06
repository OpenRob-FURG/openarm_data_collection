#!/usr/bin/env python3
"""
Teleop bimanual OpenArm com IK do MoveIt2.
Baseado na versão funcional de braço único (squeeze=embreagem, trigger=garra).

Parâmetro `arm` (both | left | right, padrão both): com left/right só esse braço
é instanciado — sem assinatura do controle do outro lado, sem IK, sem publisher
no controller dele. Os handlers de botão já ignoram controle desconhecido.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from trajectory_msgs.msg import JointTrajectoryPoint
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import threading

import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from openarm_data_collection.joint_layout import check_arm, sides_for_arm


class ArmSide:
    """Estado e IO de um braço (left/right)."""
    def __init__(self, node: Node, side: str, tf_buffer: Buffer, lock: threading.Lock):
        self.node = node
        self.side = side
        self.tf_buffer = tf_buffer
        self.lock = lock

        self.joint_names = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
        self.moveit_group_name = f"{side}_arm"
        self.ee_link_name = f"openarm_{side}_hand"
        self.finger_joint = f'openarm_{side}_finger_joint1'

        self.hardware_pos = np.zeros(7, dtype=float)
        self.current_pos = np.zeros(7, dtype=float)
        self.target_pos = np.zeros(7, dtype=float)
        self.has_hardware_state = False

        self.active = False
        self.smoothing = 0.2

        self.vr_ref_pos = None
        self.vr_ref_rot = None
        self.ee_ref_pos = None
        self.ee_ref_rot = None
        self.last_transformed_pose = None
        self.skip_first_ik_cycle = False

        self.pos_map = [0, 1, 2]
        self.pos_sign = [1.0, 1.0, 1.0]
        self.pos_gain = [1.1, 1.1, 1.1]
        self.rot_map = [0, 1, 2]
        self.rot_sign = [1.0, 1.0, 1.0]
        self.rot_gain = [1.0, 1.0, 1.0]

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(PoseStamped, f'/xr/controller_{side}/pose', self.pose_cb, qos)

        self.arm_pub = node.create_publisher(Float64MultiArray, f'/{side}_forward_position_controller/commands', qos)
        self.ik_client = node.create_client(GetPositionIK, '/compute_ik')
        self.gripper_client = ActionClient(node, FollowJointTrajectory, f'/{side}_gripper_controller/follow_joint_trajectory')
        self.gripper_goal_future = None

        self.target_pos_pub = node.create_publisher(PoseStamped, f'/debug/{side}_target_pose', qos)

    def pose_cb(self, msg):
        transformed = self.node.transform_pose_manual(msg)
        if transformed is None:
            return

        with self.lock:
            self.last_transformed_pose = transformed
            self.last_pose_time = self.node.get_clock().now()

    def joint_cb(self, msg):
        with self.lock:
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.hardware_pos[i] = msg.position[idx]
            if not self.has_hardware_state:
                self.current_pos = self.hardware_pos.copy()
                self.target_pos = self.hardware_pos.copy()
                self.has_hardware_state = True
                self.set_gripper(0.044)
                self.node.get_logger().info(f'✅ [{self.side}] Motores sincronizados!')

    def limit_joint_step(self, current, target, max_step=0.2):
        delta = target - current
        delta = np.clip(delta, -max_step, max_step)
        return current + delta

    def get_ee_pose(self, base_link_name):
        try:
            trans = self.tf_buffer.lookup_transform(base_link_name, self.ee_link_name, rclpy.time.Time())
            t = trans.transform.translation
            q = trans.transform.rotation
            pos = np.array([t.x, t.y, t.z])
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            return pos, rot
        except TransformException:
            return None, None

    def set_gripper(self, value):
        if not self.gripper_client.wait_for_server(timeout_sec=0.0):
            self.node.get_logger().warn(f'⏳ [{self.side}] Aguardando action server do gripper...')
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.finger_joint]
        point = JointTrajectoryPoint()
        point.positions = [float(value)]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 400_000_000
        goal.trajectory.points = [point]
        self.gripper_goal_future = self.gripper_client.send_goal_async(goal)
        self.gripper_goal_future.add_done_callback(self._gripper_goal_response_cb)

    def _gripper_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.node.get_logger().warn(f'⚠️ [{self.side}] Gripper goal rejeitado')
                return
            goal_handle.get_result_async()
        except Exception as e:
            self.node.get_logger().warn(f'⚠️ [{self.side}] Falha gripper: {e}')

    def call_ik_service(self, target_pos, target_rot, base_link_name):
        if not self.ik_client.wait_for_service(timeout_sec=0.01):
            return

        seed = self.target_pos.copy()
        seed[3] = max(seed[3], 0.001)  # joint4 (índice 3), evita violar lower=0.0

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.moveit_group_name
        req.ik_request.ik_link_name = self.ee_link_name
        req.ik_request.pose_stamped.header.frame_id = base_link_name
        req.ik_request.robot_state = RobotState()
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = seed.tolist()
        req.ik_request.robot_state.is_diff = False
        req.ik_request.pose_stamped.pose.position.x = target_pos[0]
        req.ik_request.pose_stamped.pose.position.y = target_pos[1]
        req.ik_request.pose_stamped.pose.position.z = target_pos[2]
        q = target_rot.as_quat()
        req.ik_request.pose_stamped.pose.orientation.x = q[0]
        req.ik_request.pose_stamped.pose.orientation.y = q[1]
        req.ik_request.pose_stamped.pose.orientation.z = q[2]
        req.ik_request.pose_stamped.pose.orientation.w = q[3]
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.nanosec = 250_000_000
        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_cb)

    def ik_response_cb(self, future):
        try:
            res = future.result()
            if res.error_code.val == 1:
                joint_names = res.solution.joint_state.name
                joint_positions = res.solution.joint_state.position
                with self.lock:
                    new_target = self.target_pos.copy()
                    for i, name in enumerate(self.joint_names):
                        if name in joint_names:
                            idx = joint_names.index(name)
                            new_target[i] = joint_positions[idx]
                    self.target_pos = self.limit_joint_step(self.target_pos, new_target)
            else:
                self.node.get_logger().warn(f'[{self.side}] IK falhou, code={res.error_code.val}')
        except Exception as e:
            self.node.get_logger().warn(f'[{self.side}] IK exception: {e}')


class BiArmTeleopIK(Node):
    def __init__(self):
        super().__init__('biarm_teleop_ik')

        self.base_link_name = "openarm_body_link0"
        self.lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_transform = None

        self.arm = check_arm(self.declare_parameter('arm', 'both').value)
        self.arms = {side: ArmSide(self, side, self.tf_buffer, self.lock)
                     for side in sides_for_arm(self.arm)}

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.btn_down_sub = self.create_subscription(String, '/xr/events/button_down', self.btn_down_cb, qos)
        self.btn_up_sub = self.create_subscription(String, '/xr/events/button_up', self.btn_up_cb, qos)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        self.timer = self.create_timer(0.01, self.control_loop)
        self.ik_timer = self.create_timer(0.033, self.ik_request_loop)
        self.check_timer = self.create_timer(1.0, self.check_tf)

        self.get_logger().info(
            f"🚀 OPENARM IK {'BIMANUAL' if self.arm == 'both' else 'BRAÇO ' + self.arm.upper()}"
            f" | braços: {list(self.arms)}")
        self.get_logger().info('⏳ Aguardando /joint_states...')

    def joint_cb(self, msg):
        for arm in self.arms.values():
            arm.joint_cb(msg)

    def check_tf(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.base_link_name, 'xr_local', rclpy.time.Time())
            self.tf_static_transform = trans
            self.check_timer.cancel()
            self.get_logger().info('✅ Referencial do VR Alinhado!')
        except Exception:
            pass

    def transform_pose_manual(self, pose_msg):
        try:
            transformed = self.tf_buffer.transform(
                pose_msg,
                self.base_link_name,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return transformed
        except TransformException as e:
            self.get_logger().warn(
                f'Falha ao transformar pose VR: {e}',
                throttle_duration_sec=2.0
            )
            return None

    def _pose_to_arrays(self, pose_msg):
        pos = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        ori = R.from_quat([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                           pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        return pos, ori

    def btn_down_cb(self, msg):
        try:
            data = json.loads(msg.data)
            btn = data.get('button', '')
            ctrl = data.get('controller', '')
            if ctrl not in self.arms:
                return
            arm = self.arms[ctrl]

            if btn == 'squeeze':
                with self.lock:
                    if arm.last_transformed_pose and arm.has_hardware_state:
                        arm.target_pos = arm.hardware_pos.copy()
                        arm.current_pos = arm.hardware_pos.copy()
                        arm.vr_ref_pos, arm.vr_ref_rot = self._pose_to_arrays(arm.last_transformed_pose)
                        ee_pos, ee_rot = arm.get_ee_pose(self.base_link_name)
                        if ee_pos is not None:
                            arm.ee_ref_pos = ee_pos
                            arm.ee_ref_rot = ee_rot
                            arm.active = True
                            arm.skip_first_ik_cycle = True
                            self.get_logger().info(f'🟢 [{ctrl}] EMBREAGEM LIGADA')
            elif btn == 'trigger':
                arm.set_gripper(0.0003)
        except json.JSONDecodeError:
            pass

    def btn_up_cb(self, msg):
        try:
            data = json.loads(msg.data)
            btn = data.get('button', '')
            ctrl = data.get('controller', '')
            if ctrl not in self.arms:
                return
            arm = self.arms[ctrl]

            if btn == 'squeeze':
                with self.lock:
                    arm.active = False
                    arm.target_pos = arm.current_pos.copy()
                self.get_logger().info(f'⏸️ [{ctrl}] CONGELADO')
            elif btn == 'trigger':
                arm.set_gripper(0.044)
        except json.JSONDecodeError:
            pass

    def ik_request_loop(self):
        for side, arm in self.arms.items():
            if not arm.active or arm.ee_ref_pos is None:
                continue
            if not arm.has_hardware_state:
                continue
            if arm.skip_first_ik_cycle:
                with self.lock:
                    arm.skip_first_ik_cycle = False
                continue

            with self.lock:
                vr_msg = arm.last_transformed_pose
            if vr_msg is None:
                continue

            vr_pos, vr_rot = self._pose_to_arrays(vr_msg)
            delta_pos = vr_pos - arm.vr_ref_pos
            delta_rot = arm.vr_ref_rot.inv() * vr_rot

            delta_pos_mapped = np.zeros(3)
            for i in range(3):
                delta_pos_mapped[arm.pos_map[i]] = delta_pos[i] * arm.pos_sign[i] * arm.pos_gain[i]

            delta_euler = delta_rot.as_euler('xyz')
            delta_euler_mapped = np.zeros(3)
            for i in range(3):
                delta_euler_mapped[arm.rot_map[i]] = delta_euler[i] * arm.rot_sign[i] * arm.rot_gain[i]
            delta_rot_mapped = R.from_euler('xyz', delta_euler_mapped)

            target_pos = arm.ee_ref_pos + delta_pos_mapped
            target_rot = delta_rot_mapped * arm.ee_ref_rot

            target_rot_viz = target_rot 
            target_rot_viz = target_rot_viz * R.from_euler('y', -np.pi/2)  # Ajuste para visualização

            q_target = target_rot_viz.as_quat()

            target_pose_msg = PoseStamped()
            target_pose_msg.header.frame_id = self.base_link_name
            target_pose_msg.pose.position.x, target_pose_msg.pose.position.y, target_pose_msg.pose.position.z = target_pos.tolist()
            target_pose_msg.pose.orientation.x, target_pose_msg.pose.orientation.y, target_pose_msg.pose.orientation.z, target_pose_msg.pose.orientation.w = q_target.tolist()

            arm.target_pos_pub.publish(target_pose_msg)

            arm.call_ik_service(target_pos, target_rot, self.base_link_name)

    def control_loop(self):
        for arm in self.arms.values():
            if not arm.has_hardware_state:
                continue
            with self.lock:
                # Só comanda com a embreagem engatada. Publicando sempre, o
                # teleop mandava current_pos (congelado na pose de startup) a
                # 100 Hz e sobrescrevia os waypoints do home_mover, que sai a
                # 20 Hz — o braço ficava preso e o home nunca acontecia. Com a
                # embreagem solta o ForwardCommandController segura o último
                # comando sozinho, e o engate ressincroniza current_pos com o
                # hardware, então não tem tranco.
                if not arm.active:
                    continue
                arm.current_pos = (1 - arm.smoothing) * arm.current_pos + arm.smoothing * arm.target_pos
                msg = Float64MultiArray()
                msg.data = arm.current_pos.tolist()
                arm.arm_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BiArmTeleopIK()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGINT ou SIGTERM
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()