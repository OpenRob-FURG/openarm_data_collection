# openarm_data_collection

Coleta de dataset para teleop **bimanual** do OpenArm via TeleopXR (WebXR / Meta
Quest), gravando no formato **LeRobot codebase v3.0**, compatível com as
convenções do `bi_openarm_follower` do OpenArm LeRobot.

## Schema do dataset

Igual ao pipeline CAN nativo (`bi_openarm_follower`), estendido para 3 câmeras:

| Feature | dtype | shape | conteúdo |
|---|---|---|---|
| `observation.state` | float32 | (16,) | `left_joint_1.pos … left_joint_7.pos, left_gripper.pos, right_joint_1.pos … right_joint_7.pos, right_gripper.pos` |
| `action` | float32 | (16,) | mesmos nomes; **alvo comandado** pelo teleop (controllers + garras) |
| `observation.images.cam_chest` | video | (480,640,3) | câmera do peito (C270 ou RealSense D435i) |
| `observation.images.left_cam_wrist` | video | (480,640,3) | webcam do pulso esquerdo |
| `observation.images.right_cam_wrist` | video | (480,640,3) | webcam do pulso direito |

- fps 30, vídeo AV1 (`libsvtav1`), `robot_type: bi_openarm_follower`.
- **Unidades:** as juntas são gravadas como o `/joint_states` publica —
  **radianos** por padrão. O dataset de referência `openarm_follower` (pipeline
  CAN) usa **graus**; para gravar igual, ver `arm_angle_scale` abaixo.
- **Diferença vs. referência de 1 braço:** aquele dataset é 8-dim
  (`joint_1.pos…gripper.pos`, `cam_chest`+`cam_wrist`). Este é bimanual (16-dim,
  3 câmeras) — mesmo pipeline e convenções, mas policy nova.

## Setup (uma vez)

```bash
# driver das webcams
sudo apt install -y ros-jazzy-v4l2-camera        # realsense2-camera já instalado

# venv Python 3.12 com rclpy (ROS) + lerobot no mesmo processo
python3.12 -m venv --system-site-packages ~/openarm_ws/.venv312
~/openarm_ws/.venv312/bin/pip install "lerobot[dataset]==0.6.1" "scipy>=1.13"

cd ~/openarm_ws && colcon build --packages-select openarm_data_collection --symlink-install
```

Por que o venv: `lerobot` exige Python ≥3.12 + NumPy ≥2; o `rclpy` do Jazzy é
Python 3.12. O `--system-site-packages` dá acesso ao `rclpy`. Não usamos
`cv_bridge` (quebra sob NumPy 2) — a decodificação de imagem é feita em
`image_utils.py`.

## Uso

Com o robô já em teleop (ros2_control + `teleop_xr` no modo ROS publicando nos
`*_joint_trajectory_controller` e `*_gripper_controller`):

```bash
source ~/openarm_ws/install/setup.bash
ros2 launch openarm_data_collection data_collection.launch.py \
    task_name:="pegar o copo vermelho" \
    dataset_root:=/home/popzap-lab/datasets/openarm_bimanual_vr
```

Sobe RealSense D435i (peito) + 2× `v4l2_camera` (pulsos) + `episode_recorder` +
`home_mover`. Verificado com hardware real (2026-09-09): 3 câmeras a ~30 fps,
episódio gravado e validado.

**Devices de câmera ficam em `config/recording_params.yaml` (bloco `cameras:`)**,
não no launch — é ali que se edita quando uma porta USB muda:

```yaml
cameras:
  chest_camera: realsense        # realsense | v4l2
  wrist_driver: usb_cam          # usb_cam (MJPG) | v4l2 (YUYV)
  chest_serial: ""                # vazio = aceita a RealSense conectada
  left_wrist_device: "/dev/v4l/by-path/...-usb-0:4:1.0-video-index0"
  right_wrist_device: "/dev/v4l/by-path/...-usb-0:10.4:1.0-video-index0"
```

⚠️ **As 2 C270 andam desconectando/reconectando sozinhas** (cabo/hub com mau
contato) e **trocam de porta USB** quando isso acontece. Se a gravação vier com
a câmera errada (ou sem imagem), confere antes de editar o yaml:
```bash
ls -l /dev/v4l/by-path/          # portas atuais
```
e tira uma captura de teste de cada `/dev/videoN` candidato (o conteúdo do
frame diz qual pulso é — ex. um objeto de referência visível só de um lado)
antes de decidir qual porta é esquerda/direita.

- As C270 têm serial de fábrica igual → **`by-id` colide, use `by-path`** (já é o default).
  O launch resolve o symlink pra `/dev/videoN` (o `usb_cam` não segue symlink).
- **RealSense sem serial fixo** por padrão — o serial já mudou entre sessões
  (outra unidade, ou o cabo reconectando diferente). Só fixe `chest_serial` se
  tiver mais de uma RealSense no barramento.
- **IMU da RealSense desligada** no launch: o driver `hid-sensor` do kernel prende
  o gyro/accel (`Permission denied` / `Hid device is busy`). Não precisamos dela.
- **Webcams via `usb_cam` + MJPG** (`wrist_driver:=usb_cam`, default). Verificado:
  as 2 C270 no mesmo hub USB2 a **30 fps limpo**. Com `v4l2_camera` (YUYV cru) as
  2 juntas no mesmo hub davam **tearing** (frames rasgados apesar do `hz`=30) —
  use `v4l2` só se separar as webcams em controladoras USB diferentes.

### Controle (Meta Quest, via `xr_buttons`)

| Botão | Evento | Ação |
|---|---|---|
| **A** (tap, controle direito) | START | começa um episódio |
| **B** (tap, controle direito) | STOP | salva o episódio, depois manda os braços pra home (`/data_collection/go_home`) |
| **X** (long-press, controle esquerdo) | DISCARD | descarta o episódio em andamento |

Episódios com menos de `min_episode_frames` (15) são descartados
automaticamente. Após o **B** há uma pausa de alguns segundos enquanto os
3 vídeos são codificados em AV1.

### Args do launch

`cameras:=false` (usar seus próprios drivers), `home_mover:=false`,
`repo_id:=org/nome`, `python:=/caminho`. `chest_camera:=`, `wrist_driver:=`,
`chest_serial:=`, `*_wrist_device:=` sobrescrevem por sessão o que já vem do
`cameras:` do yaml — normalmente não precisa passar nenhum deles.

Parâmetros finos (tópicos, nomes de junta, `arm_angle_scale`,
`gripper_scale/offset`, `sync_slop`) em `config/recording_params.yaml`.
Para gravar em **graus** como o pipeline CAN: `arm_angle_scale: 57.29578`.

## Validação

```bash
~/openarm_ws/.venv312/bin/python -m openarm_data_collection.validate_dataset \
    --root /home/popzap-lab/datasets/openarm_bimanual_vr --dump-episode 0
# ou: ros2 run openarm_data_collection validate_dataset --root <path>
```

Checa schema vs. `bi_openarm_follower`, contagem/duração/jitter dos episódios,
ranges de `state`/`action`, e (com `--dump-episode N`) grava PNGs de amostra.

## Testar o movimento gravado (sem modelo)

Prova que radiano + ordem das juntas + fiação estão certos: republica o `action`
gravado nos controllers.

```bash
# 1) checagem seca — não move nada, compara os ranges com os limites do URDF
ros2 run openarm_data_collection dataset_replay \
    --root <path> --episode 0 --dry-run

# 2) replay lento, indo devagar até o 1º frame primeiro
~/openarm_ws/.venv312/bin/python -m openarm_data_collection.dataset_replay \
    --root <path> --episode 0 --rate-scale 0.25 --goto-start
```

Aborta sozinho se o episódio exceder os limites do URDF (`--skip-limit-check`
força). Precisa do venv 3.12 pra ler o dataset — `ros2 run` funciona pro
`--dry-run`, mas o replay real é melhor pelo `.venv312/bin/python -m ...`.

## Deploy da policy

Treino é padrão: `lerobot-train` lê o dataset direto. **Inferência fica no ROS**
(não `lerobot-rollout`/`lerobot-eval`, que instanciam o robô CAN e conflitam com
o ros2_control). O runner de policy é um nó que assina `/joint_states` + câmeras,
roda `PreTrainedPolicy.from_pretrained`, e publica nos mesmos tópicos do
`dataset_replay` — passando por um safety wrapper (clip de delta + limites).
Ainda não implementado.

## Componentes

| Arquivo | Papel |
|---|---|
| `episode_recorder.py` | nó principal: sincroniza `/joint_states` + 3 câmeras, monta state/action, grava via LeRobot |
| `joint_layout.py` | layout canônico dos 16 dims + limites do URDF (fonte única: recorder, replay, runner) |
| `lerobot_writer.py` | wrapper de `LeRobotDataset` (create/resume/add_frame/save/discard/finalize), offline |
| `image_utils.py` | `sensor_msgs` → numpy RGB uint8 sem `cv_bridge` |
| `xr_buttons.py` | eventos brutos do TeleopXR → START/STOP/DISCARD |
| `home_mover.py` | leva os braços pra home **suavemente** (trajetória com ~80 waypoints, ease-in-out cosseno, lê a pose atual do `/joint_states`). Gatilho: `/data_collection/go_home` ou botão B |
| `dataset_replay.py` | republica o `action` gravado nos controllers (teste sem modelo) |
| `validate_dataset.py` | verificação pós-gravação |

## Robô: v1.0

Este robô é **OpenArm v1.0**. Bringup:

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    arm_type:=openarm_v1.0 use_fake_hardware:=false \
    right_can_interface:=can0 left_can_interface:=can1
```

Nomes das juntas e limites dos braços = iguais ao v2.0. A **garra é prismática**
(`finger_joint1`, 0–0.044 m) — no v2.0 era revoluta. `joint_layout.py` já está
com os limites do v1.

## Pendências / o que revisar

- **`joint6` (os dois braços) e as duas garras NÃO respondem** a comando via
  `*_controller/joint_trajectory` (testado 2026-09-09; j1,j2,j4,j5,j7 movem OK).
  Comando sai no formato certo — provável motor não energizado / sem torque /
  descalibrado. **Revisar antes de coletar dados** (senão essas dims ficam
  travadas no dataset).
- **ros2_control NÃO enforça limites de junta** ("Enforcing command limits is
  disabled. Command limits from URDF will be ignored") → o **safety wrapper**
  (clip de delta + limites do URDF) é obrigatório antes de rodar policy.
- **Garra em unidade crua** do `/joint_states` (metros, 0–0.044). O pipeline CAN
  de referência usa outra escala — ajuste `gripper_scale`/`gripper_offset` se for
  misturar datasets.
- **Zero/sinal das juntas:** assume-se que o `/joint_states` do ros2_control bate
  com a calibração do pipeline CAN. Conferir com um episódio de referência na
  mesma pose se for misturar datasets.
