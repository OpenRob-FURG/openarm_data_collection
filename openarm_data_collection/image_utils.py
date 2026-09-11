"""Conversão de mensagens sensor_msgs de imagem para numpy sem cv_bridge.

O cv_bridge do ROS Jazzy é compilado contra NumPy 1.x e quebra sob o NumPy 2.x
que o lerobot exige. Como o recorder roda num venv com NumPy 2.x, decodificamos
as imagens na mão: CompressedImage via cv2.imdecode, Image cru via reshape.
"""
from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

# Encodings crus que sabemos tratar -> (nº de canais, ordem já em RGB?)
_RAW_ENCODINGS = {
    "rgb8": (3, True),
    "bgr8": (3, False),
    "rgba8": (4, True),
    "bgra8": (4, False),
    "mono8": (1, True),
    "mono16": (1, True),
}


def _raw_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in _RAW_ENCODINGS:
        raise ValueError(f"encoding cru não suportado: {msg.encoding!r}")
    channels, _ = _RAW_ENCODINGS[enc]
    dtype = np.uint16 if enc == "mono16" else np.uint8
    buf = np.frombuffer(msg.data, dtype=dtype)
    arr = buf.reshape(msg.height, msg.width, channels) if channels > 1 else buf.reshape(msg.height, msg.width)
    if enc in ("rgb8", "rgba8"):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif enc in ("mono8", "mono16"):
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return arr


def image_msg_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image ou /CompressedImage -> np.ndarray HxWx3 BGR uint8."""
    if isinstance(msg, CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("falha ao decodificar CompressedImage")
        return img
    if isinstance(msg, Image):
        return _raw_to_bgr(msg)
    raise TypeError(f"tipo de mensagem de imagem inesperado: {type(msg)}")


def to_dataset_frame(msg, height: int, width: int) -> np.ndarray:
    """Converte a mensagem para o tensor esperado pelo LeRobot: HxWx3 RGB uint8,
    redimensionado para (height, width)."""
    bgr = image_msg_to_bgr(msg)
    if bgr.shape[0] != height or bgr.shape[1] != width:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
