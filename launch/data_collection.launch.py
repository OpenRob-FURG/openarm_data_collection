"""Sobe o pipeline de coleta de dataset: 3 câmeras + gravador + home_mover.

Cada câmera é RealSense ou webcam (bloco `cameras:` do yaml); as dos pulsos publicam
sempre em /{left,right}_wrist/image_raw, qualquer que seja o driver.

    ros2 launch openarm_data_collection data_collection.launch.py

Drivers, seriais e devices de câmera (chest_camera, left/right_wrist_camera,
wrist_driver, chest_serial, left/right_wrist_serial, chest_device,
left/right_wrist_device) vêm do bloco `cameras:` em config/recording_params.yaml —
é ali que se edita quando uma porta USB muda, sem precisar tocar neste arquivo. Os
args abaixo (CLI) sobrescrevem o yaml pra uma sessão específica.

Args úteis:
    arm:=both|left|right    qual braço gravar/teleoperar (padrão both = bimanual,
                            16 dims, 3 câmeras). right/left = 8 dims, só o peito + o
                            pulso daquele braço: o driver do outro pulso nem sobe, o
                            dataset vai pra ~/datasets/openarm_{right,left}_arm_vr e
                            o home_mover continua parqueando os DOIS braços no start.
    cameras:=false          não sobe os drivers de câmera (use os seus)
    task_name:="..."        sobrescreve a tarefa gravada
    dataset_root:=/path      onde gravar o dataset LeRobot
    repo_id:=org/nome        repo_id do dataset
    chest_serial:=XXHEX      serial da RealSense do peito (obrigatório se houver mais
                             de uma RealSense no launch; `rs-enumerate-devices -s`)
    left_wrist_camera:=realsense|v4l2   driver do pulso esquerdo (idem right_wrist_camera)
    left_wrist_serial:=XXHEX  serial da RealSense (D405) do pulso esquerdo
    right_wrist_serial:=XXHEX serial da RealSense (D405) do pulso direito
    left_wrist_device:=/dev/videoN   webcam do pulso esquerdo (quando v4l2)
    right_wrist_device:=/dev/videoN  webcam do pulso direito (quando v4l2)
    python:=/path/to/python  interpretador do gravador (precisa de rclpy + lerobot)
    home_mover:=false        não sobe o home_mover
    teleop:=false            não sobe o bimanual_teleop
    xr_tf:=false             não publica a TF do referencial do VR (use quando
                             já houver outro publisher desse par de frames)
    xr_x:=.. xr_yaw:=..      sobrescrevem o alinhamento do VR pra uma sessão;
                             o valor de montagem fica no bloco `xr_tf:` do yaml
                             (também xr_y, xr_z, xr_roll, xr_pitch,
                             xr_parent_frame, xr_child_frame)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from openarm_data_collection.camera_config import (
    DRIVERS,
    REALSENSE,
    WRIST_TOPIC,
    check_driver,
    check_realsense_serials,
    realsense_serial,
)
from openarm_data_collection.joint_layout import ARMS, camera_names_for_arm, check_arm

_PKG = "openarm_data_collection"


def _setup(context, *_a, **_kw):
    pkg = get_package_share_directory(_PKG)
    params = os.path.join(pkg, "config", "recording_params.yaml")

    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    arm = check_arm(cfg("arm"))
    cams_needed = camera_names_for_arm(arm)   # cam_chest + pulso(s) do braço

    recorder_overrides = {
        key: cfg(key) for key in ("task_name", "dataset_root", "repo_id") if cfg(key)
    }
    recorder_overrides["arm"] = arm

    wrist_driver = cfg("wrist_driver")

    def resolve_dev(path):
        # by-path/by-id são symlinks tipo ../../videoN. O usb_cam não segue o
        # link (tenta abrir /dev/../../videoN) -> resolvemos aqui pra /dev/videoN.
        return os.path.realpath(path) if os.path.islink(path) else path

    def webcam(ns, device, frame):
        device = resolve_dev(device)
        if wrist_driver == "usb_cam":
            # MJPG: ~10x menos banda que YUYV cru -> 2 C270 no mesmo hub USB2
            # sem tearing. Requer: sudo apt install ros-jazzy-usb-cam
            return Node(
                package="usb_cam", executable="usb_cam_node_exe",
                name="usb_cam", namespace=ns, output="screen",
                parameters=[{
                    "video_device": device,
                    "image_width": 640, "image_height": 480,
                    "framerate": 30.0,
                    "pixel_format": "mjpeg2rgb",
                    "frame_id": frame,
                }],
            )
        # v4l2_camera (já instalado): YUYV cru. Se as 2 webcams estiverem no
        # mesmo hub USB2 -> tearing; separe as portas ou use wrist_driver:=usb_cam.
        return Node(
            package="v4l2_camera", executable="v4l2_camera_node",
            name="v4l2_camera", namespace=ns, output="screen",
            parameters=[{
                "video_device": device,
                "image_size": [640, 480],
                "time_per_frame": [1, 30],
                "pixel_format": "YUYV",
                "camera_frame_id": frame,
            }],
        )
    v4l2 = webcam  # alias p/ o peito quando chest_camera=v4l2

    run_cameras = cfg("cameras").lower() in ("true", "1", "yes")

    def realsense(name, serial, extra=None, remappings=None):
        # IMU desligada de propósito: o driver hid-sensor do kernel prende o
        # gyro/accel e o librealsense não abre ("Permission denied" / "busy").
        # `rgb_camera.color_profile` vale pras D4xx com sensor RGB; a D405 não tem
        # (a cor sai do módulo de profundidade -> `depth_module.color_profile`, que
        # o chamador passa em `extra`). O driver só declara a chave que existe no
        # aparelho e ignora a outra.
        return Node(
            package="realsense2_camera", executable="realsense2_camera_node",
            name=name, namespace="/", output="screen",
            parameters=[{
                "serial_no": serial,
                "enable_color": True, "enable_depth": False,
                "enable_infra1": False, "enable_infra2": False,
                "enable_gyro": False, "enable_accel": False,
                "rgb_camera.color_profile": "640x480x30",
                **(extra or {}),
            }],
            remappings=remappings or [],
        )

    chest_driver = check_driver(cfg("chest_camera"), "chest_camera")
    wrist_camera = {
        "left": check_driver(cfg("left_wrist_camera"), "left_wrist_camera"),
        "right": check_driver(cfg("right_wrist_camera"), "right_wrist_camera"),
    }
    serial = {
        "chest": realsense_serial(cfg("chest_serial")),
        "left": realsense_serial(cfg("left_wrist_serial")),
        "right": realsense_serial(cfg("right_wrist_serial")),
    }

    # RealSense que ESTE launch vai abrir (pulso ocioso não conta). Com mais de uma
    # no barramento serial vazio deixaria o peito pegar uma D405 de pulso.
    if run_cameras:
        in_use = {}
        if chest_driver == REALSENSE:
            in_use["chest_serial"] = serial["chest"]
        for side in ("left", "right"):
            if f"{side}_cam_wrist" in cams_needed and wrist_camera[side] == REALSENSE:
                in_use[f"{side}_wrist_serial"] = serial[side]
        check_realsense_serials(in_use)

    if chest_driver == REALSENSE:
        chest_node = realsense("cam_chest", serial["chest"])
        chest_topic = "/cam_chest/color/image_raw"
    else:
        chest_node = v4l2("cam_chest", cfg("chest_device"), "chest_cam")
        chest_topic = "/cam_chest/image_raw"

    if run_cameras:
        # casa os tópicos do recorder com os drivers que este launch sobe
        recorder_overrides.setdefault("cam_chest_topic", chest_topic)
        recorder_overrides.setdefault("left_wrist_topic", WRIST_TOPIC.format(side="left"))
        recorder_overrides.setdefault("right_wrist_topic", WRIST_TOPIC.format(side="right"))

    def wrist_node(side):
        """(nó do driver, é RealSense?)"""
        if wrist_camera[side] == REALSENSE:
            ns = f"{side}_wrist"
            # A RealSense publica em /<nome>/color/image_raw; remapeia pro tópico da
            # webcam, que é o que recorder e runner assinam. Assim o driver do pulso
            # é só configuração: nada mais precisa saber qual é.
            return realsense(
                ns, serial[side], extra={"depth_module.color_profile": "640x480x30"},
                remappings=[(f"/{ns}/color/image_raw", WRIST_TOPIC.format(side=side))]), True
        return v4l2(f"{side}_wrist", cfg(f"{side}_wrist_device"), f"{side}_wrist_cam"), False

    # Só sobe o driver do pulso que o dataset vai usar: no `arm=right` a câmera
    # esquerda nem é aberta (uma USB a menos competindo no barramento).
    # Várias RealSense subindo juntas disputam a enumeração do librealsense
    # ("device busy" / não acha a câmera); escalona a abertura de cada uma.
    camera_actions = [chest_node]
    n_realsense = 1 if chest_driver == REALSENSE else 0
    for side in ("left", "right"):
        if f"{side}_cam_wrist" not in cams_needed:
            continue
        node, is_realsense = wrist_node(side)
        if is_realsense and n_realsense:
            node = TimerAction(period=3.0 * n_realsense, actions=[node])
        n_realsense += is_realsense
        camera_actions.append(node)
    camera_group = GroupAction(
        condition=IfCondition(LaunchConfiguration("cameras")),
        actions=camera_actions,
    )

    recorder = Node(
        package=_PKG, executable="episode_recorder", name="episode_recorder",
        output="screen", prefix=cfg("python"),
        parameters=[params, recorder_overrides],
    )

    home_mover = Node(
        package=_PKG, executable="home_mover", name="home_mover",
        output="screen", parameters=[params, {"arm": arm}],
        condition=IfCondition(LaunchConfiguration("home_mover")),
    )

    # Sem `prefix`: o teleop roda no python do sistema. Só o recorder precisa do
    # .venv312 (lerobot). Os dois no mesmo pacote, interpretadores diferentes.
    teleop = Node(
        package=_PKG, executable="bimanual_teleop", name="bimanual_teleop",
        output="screen", emulate_tty=True, parameters=[{"arm": arm}],
        condition=IfCondition(LaunchConfiguration("teleop")),
    )

    # Alinha o referencial do Quest com o do robô. Sem esta TF o teleop não sai
    # do lugar e só emite um warn com throttle de 2s — por isso ela mora aqui, e
    # não num processo solto de terminal.
    xr_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="xr_local_to_world", output="screen",
        arguments=[
            "--frame-id", cfg("xr_parent_frame"),
            "--child-frame-id", cfg("xr_child_frame"),
            "--x", cfg("xr_x"), "--y", cfg("xr_y"), "--z", cfg("xr_z"),
            "--roll", cfg("xr_roll"), "--pitch", cfg("xr_pitch"),
            "--yaw", cfg("xr_yaw"),
        ],
        condition=IfCondition(LaunchConfiguration("xr_tf")),
    )

    return [camera_group, recorder, home_mover, teleop, xr_tf]


def _yaml_block(key):
    """Lê um bloco de config/recording_params.yaml pra usar como default de
    argumento de launch. Editar o yaml não pede rebuild, e o CLI
    (`ros2 launch ... left_wrist_device:=...`) continua tendo prioridade."""
    import yaml
    pkg = get_package_share_directory(_PKG)
    params_path = os.path.join(pkg, "config", "recording_params.yaml")
    try:
        with open(params_path) as f:
            data = yaml.safe_load(f) or {}
        return data.get(key, {}).get("ros__parameters", {})
    except (OSError, yaml.YAMLError) as exc:
        print(f"[data_collection.launch] aviso: não li '{key}:' de "
              f"{params_path} ({exc}); usando defaults embutidos.")
        return {}


def generate_launch_description():
    default_python = os.path.join(
        os.path.expanduser("~"), "openarm_ws", ".venv312", "bin", "python")
    cams = _yaml_block("cameras")
    xr = _yaml_block("xr_tf")
    # se o yaml não tiver o bloco `cameras:` (ou falhar ao ler), cai pro último
    # mapeamento conhecido em vez de silenciosamente virar device vazio.
    bp = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{}:1.0-video-index0"
    fallback_devices = {
        "chest_device": bp.format("10.4"),
        "left_wrist_device": bp.format("4"),
        "right_wrist_device": bp.format("10.4"),
    }
    for key, dflt in fallback_devices.items():
        if not cams.get(key):
            cams[key] = dflt
    return LaunchDescription([
        DeclareLaunchArgument("arm", default_value="both", choices=list(ARMS),
                              description="both (bimanual, 16 dims) | left | right (8 dims)"),
        DeclareLaunchArgument("cameras", default_value="true"),
        DeclareLaunchArgument("task_name", default_value=""),
        DeclareLaunchArgument("dataset_root", default_value=""),
        DeclareLaunchArgument("repo_id", default_value=""),
        DeclareLaunchArgument("chest_camera",
                              default_value=str(cams.get("chest_camera", "realsense")),
                              choices=["v4l2", "realsense"]),
        DeclareLaunchArgument("wrist_driver",
                              default_value=str(cams.get("wrist_driver", "usb_cam")),
                              choices=["v4l2", "usb_cam"]),
        DeclareLaunchArgument("left_wrist_camera",
                              default_value=str(cams.get("left_wrist_camera", "v4l2")),
                              choices=list(DRIVERS)),
        DeclareLaunchArgument("right_wrist_camera",
                              default_value=str(cams.get("right_wrist_camera", "v4l2")),
                              choices=list(DRIVERS)),
        DeclareLaunchArgument("chest_serial",
                              default_value=str(cams.get("chest_serial", ""))),
        DeclareLaunchArgument("left_wrist_serial",
                              default_value=str(cams.get("left_wrist_serial", ""))),
        DeclareLaunchArgument("right_wrist_serial",
                              default_value=str(cams.get("right_wrist_serial", ""))),
        DeclareLaunchArgument("chest_device",
                              default_value=str(cams.get("chest_device", ""))),
        DeclareLaunchArgument("left_wrist_device",
                              default_value=str(cams.get("left_wrist_device", ""))),
        DeclareLaunchArgument("right_wrist_device",
                              default_value=str(cams.get("right_wrist_device", ""))),
        DeclareLaunchArgument("python", default_value=default_python),
        DeclareLaunchArgument("home_mover", default_value="true"),
        DeclareLaunchArgument("teleop", default_value="true"),
        DeclareLaunchArgument("xr_tf", default_value="true"),
        DeclareLaunchArgument("xr_parent_frame",
                              default_value=str(xr.get("parent_frame", "world"))),
        DeclareLaunchArgument("xr_child_frame",
                              default_value=str(xr.get("child_frame", "xr_local"))),
        *[DeclareLaunchArgument(f"xr_{k}", default_value=str(xr.get(k, 0.0)))
          for k in ("x", "y", "z", "roll", "pitch", "yaw")],
        OpaqueFunction(function=_setup),
    ])
