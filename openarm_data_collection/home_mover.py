"""home_mover — leva os dois braços para uma pose de home, suavemente.

Dispara quando:
  * recebe std_msgs/Empty em /data_collection/go_home (publicado pelo
    episode_recorder ao salvar um episódio), ou
  * o operador aperta B no controle XR (evento HOME do xr_buttons), se
    respond_to_xr_button=True.

Gera **uma trajetória com muitos waypoints** (um a cada `waypoint_dt`,
interpolação cosseno ease-in-out) da pose atual — lida do /joint_states — até a
home, ao longo de `move_duration_sec`. Waypoints pequenos e seguidos funcionam
mesmo com os controllers em `interpolation_method: "none"` (que é o caso), sem o
"salto" de mandar um ponto único distante.
"""
from __future__ import annotations

import math

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_DEF_LEFT = [f"openarm_left_joint{i}" for i in range(1, 8)]
_DEF_RIGHT = [f"openarm_right_joint{i}" for i in range(1, 8)]


def _ease(a: float, b: float, s: float) -> float:
    """Interpola a->b com ease-in-out cosseno (s em [0,1])."""
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * s))


class HomeMover(Node):
    def __init__(self):
        super().__init__("home_mover")
        p = self.declare_parameter

        self.left_joints = list(p("left_arm_joint_names", _DEF_LEFT).value)
        self.right_joints = list(p("right_arm_joint_names", _DEF_RIGHT).value)
        self.left_home = [float(x) for x in p("left_home_positions", [0.0] * 7).value]
        self.right_home = [float(x) for x in p("right_home_positions", [0.0] * 7).value]
        self.move_duration = float(p("move_duration_sec", 4.0).value)
        self.waypoint_dt = float(p("waypoint_dt", 0.05).value)
        self.respond_to_xr = bool(p("respond_to_xr_button", False).value)
        # o bringup sobe o braço em ZERO (reto = singular). Isso leva pra pose de
        # ready assim que o pipeline sobe, pra não travar o cotovelo no teleop.
        self.home_on_start = bool(p("home_on_start", True).value)
        self.startup_delay = float(p("startup_delay_sec", 4.0).value)

        left_topic = p("left_arm_cmd_topic",
                       "/left_joint_trajectory_controller/joint_trajectory").value
        right_topic = p("right_arm_cmd_topic",
                        "/right_joint_trajectory_controller/joint_trajectory").value

        self.left_pub = self.create_publisher(JointTrajectory, left_topic, 10)
        self.right_pub = self.create_publisher(JointTrajectory, right_topic, 10)

        self._pos: dict[str, float] = {}
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(JointState, "/joint_states", self._on_js,
                                 qos_profile_sensor_data)
        self.create_subscription(Empty, "/data_collection/go_home",
                                 lambda _msg: self.go_home(), 10)
        if self.respond_to_xr:
            from openarm_data_collection.xr_buttons import RecordingEvent, XRButtonMonitor
            self._RecordingEvent = RecordingEvent
            XRButtonMonitor(self, self._on_xr_event)

        self.get_logger().info(
            f"home_mover pronto | {self.move_duration}s, waypoint a cada "
            f"{self.waypoint_dt*1000:.0f}ms | gatilhos: /data_collection/go_home"
            f"{' + botão B (XR)' if self.respond_to_xr else ''}"
            f"{' | home automático no start' if self.home_on_start else ''}")

        if self.home_on_start:
            self._startup_timer = self.create_timer(
                self.startup_delay, self._startup_home)

    def _startup_home(self):
        self._startup_timer.cancel()
        if not self._pos:
            self.get_logger().warn("home no start: sem /joint_states ainda — tentando de novo em 3s")
            self._startup_timer = self.create_timer(3.0, self._startup_home)
            return
        self.get_logger().info("➡️  home automático no startup (tira o braço da singularidade)")
        self.go_home()

    def _on_js(self, msg: JointState):
        self._pos.update(zip(msg.name, msg.position))

    def _on_xr_event(self, event):
        if event is self._RecordingEvent.HOME:
            self.go_home()

    def _smooth_traj(self, joints, home) -> JointTrajectory | None:
        if not all(j in self._pos for j in joints):
            self.get_logger().warn("sem /joint_states para todas as juntas — home abortado")
            return None
        start = [self._pos[j] for j in joints]
        n = max(2, int(round(self.move_duration / self.waypoint_dt)))

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(joints)
        for k in range(1, n + 1):
            s = k / n
            t = s * self.move_duration
            msg.points.append(JointTrajectoryPoint(
                positions=[_ease(a, b, s) for a, b in zip(start, home)],
                velocities=[0.0] * len(joints),
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9)),
            ))
        return msg

    def go_home(self):
        lt = self._smooth_traj(self.left_joints, self.left_home)
        rt = self._smooth_traj(self.right_joints, self.right_home)
        if lt is None or rt is None:
            return
        self.left_pub.publish(lt)
        self.right_pub.publish(rt)
        self.get_logger().info(f"➡️  indo para home suavemente ({len(lt.points)} waypoints)")


def main(args=None):
    rclpy.init(args=args)
    node = HomeMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
