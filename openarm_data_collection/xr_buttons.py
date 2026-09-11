"""xr_buttons.py

Traduz eventos brutos de botão do TeleopXR em eventos semânticos de
gravação: START, HOME, DISCARD.

Mapeamento (Meta Quest, confirmado por captura ao vivo neste robô):
    A (right / button_primary)   -> START   (tap)
    B (right / button_secondary) -> HOME    (tap)
    X (left  / button_primary)   -> DISCARD (long-press, evita descarte acidental)
"""
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

from std_msgs.msg import String


class RecordingEvent(Enum):
    START = auto()
    HOME = auto()
    DISCARD = auto()


@dataclass(frozen=True)
class _ButtonSpec:
    button: str
    controller: str


_START_BUTTON = _ButtonSpec("button_primary", "right")     # A
_HOME_BUTTON = _ButtonSpec("button_secondary", "right")    # B
_DISCARD_BUTTON = _ButtonSpec("button_primary", "left")    # X (só via long-press)


class XRButtonMonitor:
    """Assina os tópicos de evento de botão do TeleopXR num node já
    existente e chama `on_event(RecordingEvent)` pra START, HOME e DISCARD.

    START/HOME disparam num tap simples (/xr/events/button_down).
    DISCARD exige long-press (/xr/events/long_press).
    """

    def __init__(self, node, on_event: Callable[[RecordingEvent], None]):
        self._node = node
        self._on_event = on_event

        node.create_subscription(
            String, "/xr/events/button_down", self._on_button_down, 10
        )
        node.create_subscription(
            String, "/xr/events/long_press", self._on_long_press, 10
        )

    def _parse(self, msg: String) -> Optional[dict]:
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            self._node.get_logger().warning(f"[xr_buttons] Payload inválido: {msg.data!r}")
            return None

    def _matches(self, payload: dict, spec: _ButtonSpec) -> bool:
        return (
            payload.get("button") == spec.button
            and payload.get("controller") == spec.controller
        )

    def _on_button_down(self, msg: String) -> None:
        payload = self._parse(msg)
        if payload is None:
            return
        if self._matches(payload, _START_BUTTON):
            self._on_event(RecordingEvent.START)
        elif self._matches(payload, _HOME_BUTTON):
            self._on_event(RecordingEvent.HOME)

    def _on_long_press(self, msg: String) -> None:
        payload = self._parse(msg)
        if payload is None:
            return
        if self._matches(payload, _DISCARD_BUTTON):
            self._on_event(RecordingEvent.DISCARD)