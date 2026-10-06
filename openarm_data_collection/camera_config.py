"""Escolha e validação das câmeras (python puro, sem ROS) — usada pelo launch.

Cada câmera (peito, pulso esquerdo, pulso direito) é uma RealSense ou uma webcam
V4L2. O que muda pro resto do pipeline é só o nome do tópico de imagem; o launch
remapeia a RealSense dos pulsos pra `/{left,right}_wrist/image_raw`, o mesmo
tópico da webcam, então recorder, runner e datasets não sabem qual driver está
por baixo.
"""

REALSENSE = "realsense"
V4L2 = "v4l2"
DRIVERS = (REALSENSE, V4L2)

WRIST_TOPIC = "/{side}_wrist/image_raw"

HOW_TO_FIND_SERIAL = "rs-enumerate-devices -s   (pacote librealsense2-utils)"


def check_driver(driver, what="câmera"):
    if driver not in DRIVERS:
        raise ValueError(f"{what}: driver inválido {driver!r}; use um de {DRIVERS}")
    return driver


def realsense_serial(serial):
    """Serial no formato que o `realsense2_camera` compara: com `_` na frente.

    O driver monta "_" + serial real de cada dispositivo e compara com o parâmetro;
    sem o `_` nada casa. Aceita com ou sem, e vazio continua vazio (= qualquer uma).
    Serial só de dígitos também escapa do YAML interpretá-lo como número.
    """
    serial = str(serial or "").strip()
    if not serial:
        return ""
    return serial if serial.startswith("_") else "_" + serial


def check_realsense_serials(in_use):
    """`in_use`: {nome da câmera: serial} só das RealSense que o launch vai subir.

    Com mais de uma RealSense no barramento, serial vazio deixa o driver pegar a
    primeira que enumerar — o peito pode acabar abrindo a D405 do pulso. Aborta
    antes de subir qualquer coisa, em vez de gravar com a câmera trocada.
    """
    if len(in_use) < 2:
        return
    missing = [name for name, serial in in_use.items() if not serial]
    if missing:
        raise ValueError(
            f"{len(in_use)} RealSense no launch ({', '.join(in_use)}), mas sem serial em: "
            f"{', '.join(missing)}. Preencha no bloco `cameras:` do yaml "
            f"(ex. right_wrist_serial) ou passe `<nome>_serial:=XXXX`. "
            f"Seriais conectados: {HOW_TO_FIND_SERIAL}")
    seen = {}
    for name, serial in in_use.items():
        if serial in seen:
            raise ValueError(
                f"{name} e {seen[serial]} têm o mesmo serial ({serial}): "
                f"cada câmera precisa do seu. Seriais: {HOW_TO_FIND_SERIAL}")
        seen[serial] = name
