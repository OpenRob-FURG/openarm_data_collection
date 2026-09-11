"""Sobe o pipeline de coleta de dataset: 3 câmeras + gravador + home_mover.

    ros2 launch openarm_data_collection data_collection.launch.py

Devices/portas de câmera (chest_camera, wrist_driver, chest_serial,
chest_device, left/right_wrist_device) vêm do bloco `cameras:` em
config/recording_params.yaml — é ali que se edita quando uma porta USB muda,
sem precisar tocar neste arquivo. Os args abaixo (CLI) sobrescrevem o yaml pra
uma sessão específica.

Args úteis:
    cameras:=false          não sobe os drivers de câmera (use os seus)
    task_name:="..."        sobrescreve a tarefa gravada
    dataset_root:=/path      onde gravar o dataset LeRobot
    repo_id:=org/nome        repo_id do dataset
    chest_serial:=XXHEX      serial da RealSense do peito (se houver mais de uma)
    left_wrist_device:=/dev/videoN   webcam do pulso esquerdo
    right_wrist_device:=/dev/videoN  webcam do pulso direito
    python:=/path/to/python  interpretador do gravador (precisa de rclpy + lerobot)
    home_mover:=false        não sobe o home_mover
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PKG = "openarm_data_collection"


def _setup(context, *_a, **_kw):
    pkg = get_package_share_directory(_PKG)
    params = os.path.join(pkg, "config", "recording_params.yaml")

    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    recorder_overrides = {
        key: cfg(key) for key in ("task_name", "dataset_root", "repo_id") if cfg(key)
    }

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

    if cfg("chest_camera") == "realsense":
        # IMU desligada de propósito: o driver hid-sensor do kernel prende o
        # gyro/accel e o librealsense não abre ("Permission denied" / "busy").
        chest_node = Node(
            package="realsense2_camera", executable="realsense2_camera_node",
            name="cam_chest", namespace="/", output="screen",
            parameters=[{
                "serial_no": cfg("chest_serial"),
                "enable_color": True, "enable_depth": False,
                "enable_infra1": False, "enable_infra2": False,
                "enable_gyro": False, "enable_accel": False,
                "rgb_camera.color_profile": "640x480x30",
            }],
        )
        chest_topic = "/cam_chest/color/image_raw"
    else:
        chest_node = v4l2("cam_chest", cfg("chest_device"), "chest_cam")
        chest_topic = "/cam_chest/image_raw"

    if run_cameras:
        # casa os tópicos do recorder com os drivers que este launch sobe
        recorder_overrides.setdefault("cam_chest_topic", chest_topic)
        recorder_overrides.setdefault("left_wrist_topic", "/left_wrist/image_raw")
        recorder_overrides.setdefault("right_wrist_topic", "/right_wrist/image_raw")

    camera_group = GroupAction(
        condition=IfCondition(LaunchConfiguration("cameras")),
        actions=[
            chest_node,
            v4l2("left_wrist", cfg("left_wrist_device"), "left_wrist_cam"),
            v4l2("right_wrist", cfg("right_wrist_device"), "right_wrist_cam"),
        ],
    )

    recorder = Node(
        package=_PKG, executable="episode_recorder", name="episode_recorder",
        output="screen", prefix=cfg("python"),
        parameters=[params, recorder_overrides],
    )

    home_mover = Node(
        package=_PKG, executable="home_mover", name="home_mover",
        output="screen", parameters=[params],
        condition=IfCondition(LaunchConfiguration("home_mover")),
    )

    return [camera_group, recorder, home_mover]


def _camera_defaults():
    """Lê o bloco `cameras:` de config/recording_params.yaml. Editar as portas
    de câmera é só mexer nesse yaml — não precisa tocar neste arquivo nem
    rebuildar. CLI (`ros2 launch ... left_wrist_device:=...`) ainda tem
    prioridade sobre o yaml."""
    import yaml
    pkg = get_package_share_directory(_PKG)
    params_path = os.path.join(pkg, "config", "recording_params.yaml")
    try:
        with open(params_path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("cameras", {}).get("ros__parameters", {})
    except (OSError, yaml.YAMLError) as exc:
        print(f"[data_collection.launch] aviso: não li 'cameras:' de "
              f"{params_path} ({exc}); usando defaults embutidos.")
        return {}


def generate_launch_description():
    default_python = os.path.join(
        os.path.expanduser("~"), "openarm_ws", ".venv312", "bin", "python")
    cams = _camera_defaults()
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
        DeclareLaunchArgument("chest_serial",
                              default_value=str(cams.get("chest_serial", ""))),
        DeclareLaunchArgument("chest_device",
                              default_value=str(cams.get("chest_device", ""))),
        DeclareLaunchArgument("left_wrist_device",
                              default_value=str(cams.get("left_wrist_device", ""))),
        DeclareLaunchArgument("right_wrist_device",
                              default_value=str(cams.get("right_wrist_device", ""))),
        DeclareLaunchArgument("python", default_value=default_python),
        DeclareLaunchArgument("home_mover", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
