"""home_mover — leva os braços para uma pose de home, suavemente.

Parâmetro `arm` (both | left | right, padrão both) diz qual braço o pipeline usa:

  * home no STARTUP leva SEMPRE os dois: tira o braço ocioso da singularidade (o
    bringup sobe em zero = reto) e deixa ele na mesma pose de repouso das gravações
    bimanuais, que aparece no campo da câmera do peito;
  * home pós-episódio / botão B / /data_collection/go_home leva só o braço ativo —
    o outro não se mexe durante a coleta, então não há o que reposicionar.

Dispara quando:
  * recebe std_msgs/Empty em /data_collection/go_home (publicado pelo
    episode_recorder ao salvar um episódio), ou
  * o operador aperta B no controle XR (evento HOME do xr_buttons), se
    respond_to_xr_button=True.

Interpola da pose atual — lida do /joint_states — até a home com ease-in-out
cosseno, ao longo de `move_duration_sec`, em passos de `waypoint_dt`.

Dois modos de saída (`arm_command_mode`):

  * `forward_position` (padrão): o bringup sobe só os
    `*_forward_position_controller`, que aceitam std_msgs/Float64MultiArray em
    `/{side}_forward_position_controller/commands`. Esse controller **não
    interpola**: cada mensagem é um degrau. Por isso os waypoints são
    publicados um a um num timer de `waypoint_dt`, virando rampa.
  * `trajectory`: manda uma JointTrajectory única, com todos os waypoints, para
    `/{side}_joint_trajectory_controller/joint_trajectory`. Só funciona se os
    JTC do braço estiverem spawnados (não é o caso do bringup atual).
"""
from __future__ import annotations

import math

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_data_collection.joint_layout import check_arm, sides_for_arm

_DEF_LEFT = [f"openarm_left_joint{i}" for i in range(1, 8)]
_DEF_RIGHT = [f"openarm_right_joint{i}" for i in range(1, 8)]

_MODES = ("forward_position", "trajectory")
_DEF_TOPIC = {
    "forward_position": "/{side}_forward_position_controller/commands",
    "trajectory": "/{side}_joint_trajectory_controller/joint_trajectory",
}


def _ease(a: float, b: float, s: float) -> float:
    """Interpola a->b com ease-in-out cosseno (s em [0,1])."""
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * s))


class HomeMover(Node):
    def __init__(self):
        super().__init__("home_mover")
        p = self.declare_parameter

        self.arm = check_arm(p("arm", "both").value)
        self.active_sides = sides_for_arm(self.arm)
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

        self.mode = str(p("arm_command_mode", "forward_position").value)
        if self.mode not in _MODES:
            self.get_logger().error(
                f"arm_command_mode='{self.mode}' inválido, usando 'forward_position' "
                f"(válidos: {', '.join(_MODES)})")
            self.mode = "forward_position"

        # topico vazio => default do modo escolhido
        left_topic = (p("left_arm_cmd_topic", "").value
                      or _DEF_TOPIC[self.mode].format(side="left"))
        right_topic = (p("right_arm_cmd_topic", "").value
                       or _DEF_TOPIC[self.mode].format(side="right"))

        msg_type = Float64MultiArray if self.mode == "forward_position" else JointTrajectory
        self.left_pub = self.create_publisher(msg_type, left_topic, 10)
        self.right_pub = self.create_publisher(msg_type, right_topic, 10)
        self._topics = {"left": left_topic, "right": right_topic}

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

        # estado do streaming (só usado no modo forward_position)
        self._stream_timer = None
        self._stream: dict[str, list[list[float]]] = {}
        self._stream_idx = 0

        self.get_logger().info(
            f"home_mover pronto | arm={self.arm} | modo={self.mode} | {self.move_duration}s, waypoint a cada "
            f"{self.waypoint_dt*1000:.0f}ms | {left_topic} + {right_topic}"
            f" | gatilhos: /data_collection/go_home"
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
        self.go_home(todos=True)   # inclui o braço ocioso: ver docstring

    def _on_js(self, msg: JointState):
        self._pos.update(zip(msg.name, msg.position))

    def _on_xr_event(self, event):
        if event is self._RecordingEvent.HOME:
            self.go_home()

    def _waypoints(self, joints, home, side="") -> list[list[float]] | None:
        """Lista de posições (joint1..joint7) da pose atual até a home."""
        if not all(j in self._pos for j in joints):
            faltando = [j for j in joints if j not in self._pos]
            self.get_logger().warn(
                f"sem /joint_states para {faltando} [{side}] — esse braço não vai pra home "
                f"(confira left/right_arm_joint_names no yaml e o que "
                f"o bringup publica)")
            return None
        start = [self._pos[j] for j in joints]
        n = max(2, int(round(self.move_duration / self.waypoint_dt)))
        return [[_ease(a, b, k / n) for a, b in zip(start, home)]
                for k in range(1, n + 1)]

    def _check_subscribers(self, sides) -> bool:
        """Avisa alto se ninguém escuta — sintoma clássico de 'o braço não mexe'."""
        ok = True
        pubs = {"left": self.left_pub, "right": self.right_pub}
        for side in sides:
            pub = pubs[side]
            if pub.get_subscription_count() == 0:
                self.get_logger().error(
                    f"ninguém inscrito em {self._topics[side]} — o controller do braço "
                    f"[{side}] não está ativo (ou arm_command_mode está errado). "
                    f"O home não vai sair do lugar.")
                ok = False
        return ok

    def go_home(self, todos: bool = False):
        """Leva pra home o braço ativo (`todos=True`: os dois, usado no startup)."""
        sides = ("left", "right") if todos else self.active_sides
        spec = {"left": (self.left_joints, self.left_home),
                "right": (self.right_joints, self.right_home)}
        wps = {}
        for side in sides:
            wp = self._waypoints(*spec[side], side=side)
            if wp is None:
                if side in self.active_sides:
                    return           # o braço que vai gravar não pode ir pra home: aborta
                continue             # braço ocioso sem juntas (ex.: bringup de um braço só): segue sem ele
            wps[side] = wp
        if not wps:
            return
        self._check_subscribers(tuple(wps))
        n = len(next(iter(wps.values())))

        if self.mode == "trajectory":
            pubs = {"left": self.left_pub, "right": self.right_pub}
            for side, wp in wps.items():
                pubs[side].publish(self._as_traj(spec[side][0], wp))
            self.get_logger().info(
                f"➡️  indo para home suavemente ({n} waypoints) | braços: {list(sides)}")
            return

        # forward_position: rampa publicada waypoint a waypoint
        if self._stream_timer is not None:
            self._stream_timer.cancel()
            self._stream_timer = None
        self._stream = wps
        self._stream_idx = 0
        self._stream_timer = self.create_timer(self.waypoint_dt, self._stream_step)
        self.get_logger().info(
            f"➡️  indo para home suavemente ({n} waypoints em "
            f"{self.move_duration:.1f}s, streaming) | braços: {list(sides)}")

    def _stream_step(self):
        n = len(next(iter(self._stream.values())))
        if self._stream_idx >= n:
            self._stream_timer.cancel()
            self._stream_timer = None
            self.get_logger().info("✅ home concluído")
            return
        pubs = {"left": self.left_pub, "right": self.right_pub}
        for side, wp in self._stream.items():
            pubs[side].publish(Float64MultiArray(data=wp[self._stream_idx]))
        self._stream_idx += 1

    def _as_traj(self, joints, waypoints) -> JointTrajectory:
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(joints)
        n = len(waypoints)
        for k, pos in enumerate(waypoints, start=1):
            t = (k / n) * self.move_duration
            msg.points.append(JointTrajectoryPoint(
                positions=list(pos),
                velocities=[0.0] * len(joints),
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9)),
            ))
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = HomeMover()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGINT ou SIGTERM
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
