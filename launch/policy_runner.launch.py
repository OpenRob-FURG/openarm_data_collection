"""Sobe o policy_runner com o interpretador certo.

O nó importa `lerobot`, que só existe no `.venv312` — o python do sistema (que é
quem o `ros2 run` usa, via shebang do console script) não serve. Este launch faz
o mesmo truque do `episode_recorder` no data_collection.launch.py: `prefix` com o
python do venv.

    # seco — não publica nada nos controllers
    ros2 launch openarm_data_collection policy_runner.launch.py \\
        checkpoint:=$HOME/models/act_lego_v1_100k dry_run:=true

    # ao vivo, devagar (continua DESARMADO até você mandar o enable)
    ros2 launch openarm_data_collection policy_runner.launch.py \\
        checkpoint:=$HOME/models/act_lego_v1_100k max_step:=0.015 ramp_seconds:=5
    ros2 topic pub --once /policy_runner/enable std_msgs/Bool "{data: true}"

Sem launch, o equivalente em uma linha:
    ros2 run --prefix $HOME/openarm_ws/.venv312/bin/python \\
        openarm_data_collection policy_runner --checkpoint <path> --dry-run
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PKG = "openarm_data_collection"

# args que viram flags de linha de comando do nó (nome do launch -> flag)
_PASSTHROUGH = {
    "checkpoint": "--checkpoint",
    "arm": "--arm",
    "device": "--device",
    "task": "--task",
    "max_step": "--max-step",
    "gripper_max_step": "--gripper-max-step",
    "ramp_seconds": "--ramp-seconds",
    "obs_timeout": "--obs-timeout",
    "n_action_steps": "--n-action-steps",
    "torch_threads": "--torch-threads",
    "temporal_ensemble": "--temporal-ensemble",
    "cam_chest_topic": "--cam-chest-topic",
    "left_wrist_topic": "--left-wrist-topic",
    "right_wrist_topic": "--right-wrist-topic",
}
# args booleanos -> flag sem valor, só quando "true"
_FLAGS = {
    "dry_run": "--dry-run",
    "auto_arm": "--auto-arm",
}


def _setup(context, *_a, **_kw):
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    if not cfg("checkpoint"):
        raise RuntimeError(
            "checkpoint:= é obrigatório — aponte pra pasta pretrained_model/ "
            "(ex.: checkpoint:=$HOME/models/act_lego_v1_100k)")

    argv = []
    for name, flag in _PASSTHROUGH.items():
        value = cfg(name)
        if value:
            argv += [flag, value]
    for name, flag in _FLAGS.items():
        if cfg(name).lower() in ("true", "1", "yes"):
            argv.append(flag)
    # Tri-estado explícito: o nó agora tem default DESLIGADO, e sem o par
    # afirmativo aqui um "freeze_left:=true" explícito virava no-op silencioso
    # (nunca era passado pro nó, que ficava no próprio default).
    if cfg("freeze_left").lower() in ("true", "1", "yes"):
        argv.append("--freeze-left")
    elif cfg("freeze_left").lower() in ("false", "0", "no"):
        argv.append("--no-freeze-left")

    return [Node(
        package=_PKG, executable="policy_runner", name="policy_runner",
        output="screen", prefix=cfg("python"), arguments=argv,
    )]


def generate_launch_description():
    default_python = os.path.join(
        os.path.expanduser("~"), "openarm_ws", ".venv312", "bin", "python")

    return LaunchDescription([
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="pasta pretrained_model/ do checkpoint (obrigatório)"),
        DeclareLaunchArgument("python", default_value=default_python,
                              description="interpretador do nó (precisa de rclpy + lerobot)"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="true = roda a policy mas não publica nos controllers"),
        DeclareLaunchArgument("auto_arm", default_value="false",
                              description="true = já começa armado (cuidado)"),
        DeclareLaunchArgument("arm", default_value="both",
                              description="both (16 dims) | right | left (checkpoint unimanual, 8 dims)"),
        DeclareLaunchArgument("freeze_left", default_value="false",
                              description="ignora as 8 dims do braço esquerdo — ligue só "
                                          "se a tarefa sabidamente não usa o braço esquerdo"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("task", default_value=""),
        DeclareLaunchArgument("max_step", default_value=""),
        DeclareLaunchArgument("gripper_max_step", default_value=""),
        DeclareLaunchArgument("ramp_seconds", default_value=""),
        DeclareLaunchArgument("obs_timeout", default_value=""),
        DeclareLaunchArgument("n_action_steps", default_value="",
                              description="acoes executadas antes de re-planejar; 10 recomendado p/ SmolVLA"),
        DeclareLaunchArgument("torch_threads", default_value="2"),
        DeclareLaunchArgument(
            "temporal_ensemble", default_value="0.01",
            description="temporal ensembling do ACT; vazio desliga (= malha aberta por 3.3s)"),
        # vazios = usa o default do nó, que já casa com os tópicos do
        # data_collection.launch.py
        DeclareLaunchArgument("cam_chest_topic", default_value=""),
        DeclareLaunchArgument("left_wrist_topic", default_value=""),
        DeclareLaunchArgument("right_wrist_topic", default_value=""),
        OpaqueFunction(function=_setup),
    ])
