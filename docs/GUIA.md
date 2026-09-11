# Guia — pipeline de coleta de dataset OpenArm (v1.0 · bimanual · TeleopXR)

Estado em 2026-09-09. Cobre: o que mudou, o que revisar, e como tudo se encaixa.

---

## 1. O que foi mudado

### Pacote `src/openarm_data_collection/` (reescrito)

| Arquivo | Antes | Agora |
|---|---|---|
| `episode_recorder.py` | 1 câmera, `position[:16]` cego, `btn_id` 1/2/3, sem finalize | 3 câmeras sincronizadas, state/action **por nome de junta** (layout `bi_openarm_follower`), 4 subs de comando (2 braços + 2 garras), `xr_buttons` semântico, descarte de episódio curto, `finalize()` no shutdown, contador de episódios |
| `lerobot_writer.py` | API antiga, tentava o Hub | `LeRobotDataset` v3.0 `create`/`resume`, **offline** (`HF_HUB_OFFLINE`), imagens HxWx3 uint8 |
| `joint_layout.py` | — (novo) | **fonte única** do vetor de 16 dims (feature ↔ nome ROS) + limites do URDF **v1.0**. Usado por recorder, replay e o futuro runner |
| `image_utils.py` | — (novo) | `sensor_msgs` → numpy RGB uint8 **sem `cv_bridge`** (que quebra sob NumPy 2) |
| `home_mover.py` | **vazio** (quebrava o entry point) | leva os braços pra home ao receber `/data_collection/go_home` (ou botão B) |
| `dataset_replay.py` | — (novo) | republica o `action` gravado nos controllers — testa unidade/ordem/fiação **sem modelo**; aborta se passar dos limites do URDF |
| `validate_dataset.py` | — (novo) | checa schema vs. `bi_openarm_follower`, fps/jitter, ranges, dump de PNGs |
| `xr_buttons.py` | já existia, ocioso | agora é usado pelo recorder (A=START, B=STOP, X long=DISCARD) |
| `config/recording_params.yaml` | 3 tópicos que o código ignorava | params reais do recorder + home_mover, sem `/**` (quebrava override do launch) |
| `launch/data_collection.launch.py` | só o recorder | RealSense (peito) + 2× webcam + recorder + home_mover; args `chest_camera`, `wrist_driver`, devices, `task_name`, etc. |
| `setup.py` / `package.xml` | não instalava `config/`, faltava `trajectory_msgs` | corrigido; entry points novos (`dataset_replay`, `validate_dataset`) |

### Ambiente novo: `~/openarm_ws/.venv312`

`episode_recorder` importa `rclpy` (ROS Jazzy / Python 3.12) **e** `lerobot`
(Python ≥3.12, NumPy ≥2) no mesmo processo. O `~/openarm_ws/.venv` é 3.14 → `rclpy`
não carrega. Solução:

```bash
python3.12 -m venv --system-site-packages ~/openarm_ws/.venv312
~/openarm_ws/.venv312/bin/pip install "lerobot[dataset]==0.6.1" "scipy>=1.13"
```

O launch chama o gravador com esse interpretador (`prefix=`). **Não** instalar
`cv_bridge` (quebra sob NumPy 2).

### Descobertas de hardware corrigidas no launch

- **Robô é v1.0** — garra prismática (0–0.044 m), não revoluta. `joint_layout.py`
  já tem os limites do v1.
- **RealSense**: IMU desligada (`enable_gyro/accel:=false`) — o `hid-sensor` do
  kernel prende o gyro e o librealsense não abre.
- **Webcams**: as 3 C270 têm serial de fábrica igual → `by-id` colide → launch
  usa `by-path` **e resolve o symlink** pra `/dev/videoN` (o `usb_cam` não segue
  symlink). Default `wrist_driver:=usb_cam` (MJPG) — 2 C270 no mesmo hub USB2 a
  30 fps limpo. Com `v4l2_camera` (YUYV) dava tearing.

---

## 2. O que você precisa revisar / decidir

| # | Item | Por quê |
|---|---|---|
| 1 | **`joint6` (2 braços) e as 2 garras não movem** | Testado: comando sai certo, motor não responde. Checar energia / torque / calibração desses motores (j6 e garra = DM4310). Se ficarem travados, essas 4 dims do dataset ficam constantes. |
| 2 | **`arm_angle_scale` — radianos ou graus?** | Grava radianos (como o `/joint_states`). O dataset de referência CAN usa graus. Pra policy nova bimanual tanto faz; pra bater com dados CAN: `arm_angle_scale: 57.29578` no yaml. |
| 3 | **Garra: escala** | Grava metros crus (0–0.044). Se misturar com dados do pipeline CAN, ajustar `gripper_scale`/`gripper_offset`. |
| 4 | **`left_home_positions` / `right_home_positions`** | Hoje tudo 0.0 no yaml. Definir a pose de home real (radianos, joint_1..7). |
| 5 | **Safety wrapper (ainda não feito)** | O `ros2_control` **não enforça limites de junta** ("Command limits from URDF will be ignored"). Antes de rodar qualquer policy, precisa de um nó que satura Δ por passo + trava nos limites. |
| 6 | **Estrutura do repo** | `openarm_ws` e `openarm_data_collection` **não são git** — sem histórico. `teleop_xr` está fora de `src/`. Vale um `git init` + `.gitignore`. |
| 7 | **Tópicos dos testes lint** | `ament_flake8`/`pep257` reclamam do estilo (aspas duplas, docstrings) — já reclamavam do código pré-existente; não mexi. |

---

## 3. Pipeline final — como tudo funciona

### 3.1 Schema do dataset (LeRobot codebase v3.0)

```
observation.state   float32 (16,)   [left_joint_1..7.pos, left_gripper.pos,
action              float32 (16,)    right_joint_1..7.pos, right_gripper.pos]
observation.images.cam_chest         video 480x640x3   RealSense D435i (peito)
observation.images.left_cam_wrist    video 480x640x3   C270 pulso esquerdo
observation.images.right_cam_wrist   video 480x640x3   C270 pulso direito
```

- fps 30, vídeo AV1, `robot_type: bi_openarm_follower`, local (sem Hub).
- `action` = **alvo comandado** pelo teleop (não o próximo estado).
- Mesmas convenções do `bi_openarm_follower` do OpenArm LeRobot; a referência de
  1 braço (`romulo-maciel/new_openarm_lift_fanta_bottle_...`) é 8-dim — este é
  bimanual 16-dim, mesma família, policy nova.

### 3.2 Fluxo de gravação

```
TeleopXR (WebXR/Quest) ──► ros2_control (v1.0, can0/can1)
     │  publica em                    │  publica
     ▼                                ▼
/*_joint_trajectory_controller   /joint_states (500 Hz)
/*_gripper_controller                  │
     │                                 │
     └──────────┐          ┌───────────┘
                ▼          ▼
        episode_recorder (venv312)
         ├─ ApproximateTimeSynchronizer: /joint_states + 3 câmeras (slop 33 ms)
         ├─ monta state (16) por nome de junta · action = último comando
         ├─ botões:  A=START   B=STOP+salva   X(long)=DISCARD
         └─► LeRobotDataset  →  /home/popzap-lab/datasets/openarm_bimanual_vr
                                 (parquet + 3 mp4 AV1 por episódio)
```

### 3.3 Rodar

```bash
# 1) robô (v1.0, hardware real)
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    arm_type:=openarm_v1.0 use_fake_hardware:=false \
    right_can_interface:=can0 left_can_interface:=can1

# 2) TeleopXR no modo ROS (seu comando de sempre)

# 3) coleta: câmeras + gravador + home_mover
source ~/openarm_ws/install/setup.bash
ros2 launch openarm_data_collection data_collection.launch.py \
    task_name:="pegar o copo vermelho" \
    dataset_root:=/home/popzap-lab/datasets/openarm_bimanual_vr
```

Botões no Quest: **A** começa, **B** salva + manda pra home, **X (segurar)**
descarta. Episódio < 15 frames é descartado sozinho. Depois do B há ~4 s de
encode dos 3 vídeos.

### 3.4 Validar e testar movimento (sem modelo)

```bash
# schema, fps/jitter, ranges, PNGs de amostra
~/openarm_ws/.venv312/bin/python -m openarm_data_collection.validate_dataset \
    --root /home/popzap-lab/datasets/openarm_bimanual_vr --dump-episode 0

# checagem seca do replay: não move nada, compara ranges com limites do URDF
ros2 run openarm_data_collection dataset_replay --root <path> --episode 0 --dry-run

# replay real, 4x mais lento, indo devagar até o 1º frame
~/openarm_ws/.venv312/bin/python -m openarm_data_collection.dataset_replay \
    --root <path> --episode 0 --rate-scale 0.25 --goto-start
```

### 3.5 Deploy da policy (a implementar)

```
dataset ──► lerobot-train (padrão) ──► checkpoint
                                          │
                                          ▼
                              policy_runner (ROS, a fazer)
                               ├─ assina /joint_states + 3 câmeras
                               ├─ PreTrainedPolicy.from_pretrained → action (16)
                               └─► safety_wrapper (a fazer) ─► *_controller/joint_trajectory
```

**Não** usar `lerobot-rollout`/`lerobot-eval` — instanciam o robô CAN e brigam
com o `ros2_control`. Fica tudo em ROS, mesma interface da gravação.

---

## 4. Checklist até o primeiro dataset utilizável

- [ ] Resolver `joint6` + garras (item 2.1)
- [ ] `sudo apt install ros-jazzy-usb-cam` ✔ (feito)
- [ ] `ros-jazzy-v4l2-camera` ✔ (feito) — fallback
- [x] Pulso esq/dir confirmado: esquerda = `usb-0:9.2` (video4), direita = `usb-0:9.1` (video2, vê o Sicredi)
- [ ] Definir `left_home_positions` / `right_home_positions`
- [ ] Gravar 2–3 episódios → `validate_dataset` → `dataset_replay --dry-run` → replay lento
- [ ] Escrever `safety_wrapper` + `policy_runner`
- [ ] `git init` no workspace
