"""policy_runner — roda uma policy do LeRobot no robô, dentro do ROS.

Fecha o laço: /joint_states + 3 câmeras -> policy -> comandos nos controllers.
É o par de inferência do `episode_recorder`: monta a observação EXATAMENTE como
ela foi gravada (mesmo `to_dataset_frame`, mesma ordem de 16 dims, mesmo sync),
senão o modelo recebe entrada fora da distribuição de treino.

NÃO use `lerobot-rollout`/`lerobot-eval`: eles instanciam o robô pelo backend CAN
do LeRobot e brigam com o ros2_control, que já está segurando as juntas. (E o
backend `bi_openarm_follower` trabalha em GRAUS, enquanto este dataset foi
gravado em RADIANOS — a policy sairia com erro de fator 57.)

QUALQUER POLICY DO LEROBOT
--------------------------
O carregamento é genérico (`get_policy_class` + `make_pre_post_processors` +
`predict_action`), então serve ACT, SmolVLA, pi0, etc. Testado com ACT e SmolVLA.
O que o nó resolve sozinho por policy:

  * **renomeação de features** — se o checkpoint salvou um
    `RenameObservationsProcessorStep` (a SmolVLA renomeia cam_chest->camera1),
    ele é lido do preprocessor e as imagens são alimentadas pelos nomes de ORIGEM,
    que é o que o pipeline espera;
  * **`--temporal-ensemble`** só é aplicado se a policy for ACT (é implementação
    do ACT); nas outras é ignorado com aviso;
  * **`--n-action-steps`** funciona em qualquer policy com fila de ações.

  * **`--arm right|left`** para checkpoint unimanual (ação/estado de 8 dims): a
    policy controla só os slots daquele braço, o outro não recebe comando e só
    as câmeras que ela consome são assinadas. Se a dimensão da ação não bater com
    `--arm`, o nó aborta na partida com a sugestão certa.

O que você precisa conferir ao trocar de policy:
  * a dimensão da ação tem que bater com `--arm` (16 = both, 8 = right/left);
  * `--task` precisa ser a string EXATA do dataset de treino — policies com VLM
    (SmolVLA, pi0) tokenizam esse texto e condicionam nele; o ACT ignora;
  * deps extras (SmolVLA precisa de `transformers` e `num2words` no venv, e baixa
    o VLM base para o cache do HuggingFace).

Saída (igual ao caminho que o bimanual_teleop usou na gravação):
  * braço -> /{side}_forward_position_controller/commands  (Float64MultiArray)
  * garra -> /{side}_gripper_controller/joint_trajectory   (JointTrajectory)

SEGURANÇA — o ros2_control NÃO enforça limites de junta, então quem segura é
este nó:
  * clamp em URDF_LIMITS_RAD (joint_layout) em todo comando;
  * `--max-step` limita quanto cada junta anda por tick (o ForwardCommandController
    não interpola: uma mensagem só é um DEGRAU);
  * começa DESARMADO; só move depois de `--auto-arm` ou de um True em
    /policy_runner/enable;
  * ao armar, faz rampa suave do estado medido até a 1ª ação (`--ramp-seconds`);
  * watchdog: sem imagem/joint_states fresco, desarma e segura a pose;
  * `--freeze-left` (padrão DESLIGADO) ignora as 8 dims do braço esquerdo
    — ligue para uma tarefa que sabidamente não usa o braço esquerdo.

Nada disso protege contra bater na mesa ou no próprio robô: colisão não existe
nem nos limites de junta nem no rate limit. O botão de emergência é o backup.

Uso (venv 3.12, igual ao recorder — o python do sistema não tem lerobot):
    # seco: roda a policy mas não publica nos controllers
    ros2 launch openarm_data_collection policy_runner.launch.py \
        checkpoint:=$HOME/models/act_lego_v1_100k dry_run:=true

    # ao vivo (continua DESARMADO até o enable)
    ros2 launch openarm_data_collection policy_runner.launch.py \
        checkpoint:=$HOME/models/act_lego_v1_100k max_step:=0.04 ramp_seconds:=5
    ros2 topic pub --once /policy_runner/enable std_msgs/Bool "{data: true}"

    # desarmar sem matar o processo
    ros2 topic pub --once /policy_runner/enable std_msgs/Bool "{data: false}"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_data_collection.image_utils import to_dataset_frame
from openarm_data_collection.joint_layout import (
    ARMS, URDF_LIMITS_RAD, build_slots, check_arm, sides_for_arm,
)

LOG_PERIOD_S = 1.0   # intervalo do log periódico (em tempo, não em ticks)


def _enable_temporal_ensembling(policy, coeff: float, logger=None) -> None:
    """Liga o temporal ensembling do ACT (Alg. 2 do paper) em tempo de inferência.

    Sem isto o ACT roda em malha aberta: `select_action` só chama a rede quando a
    fila de `n_action_steps` esvazia — com n_action_steps=100 a 30 Hz são 3,3 s
    sem olhar para a observação, que é o sintoma de "não corrige". Ligado, ele
    re-prediz o chunk inteiro a cada tick e mistura as predições sobrepostas com
    pesos exp(-coeff*i). Custa ~16 ms/tick em vez de ~3 ms.

    Medido offline contra episódios reais (ep30): erro médio 0.0248 -> 0.0176 rad
    e erro máximo 0.2133 -> 0.1110. O ganho em malha fechada tende a ser maior,
    porque o teste offline sempre alimenta observações de uma demo perfeita.
    """
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    policy.config.temporal_ensemble_coeff = coeff
    policy.temporal_ensembler = ACTTemporalEnsembler(coeff, policy.config.chunk_size)
    policy.reset()
    if logger:
        logger.info(f"temporal ensembling LIGADO (coeff={coeff}) — re-prediz a cada tick")


def _load_policy(ckpt: str, device: str, torch_threads: int = 2):
    """Carrega a policy + os processors de normalização salvos no checkpoint."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # PyTorch abre 1 thread por core (24 no zap). A inferência roda na GPU e o
    # trabalho de CPU aqui é só normalizar 3 imagens — 24 threads é contenção pura
    # e MATA DE FOME os drivers de câmera e o ros2_control na mesma máquina
    # (medido: o runner sozinho em 1488% de CPU, câmeras caindo de 30 para 5 Hz e
    # o laço de controle a 12 Hz em vez de 30). Tem que vir antes de qualquer op.
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
    except ModuleNotFoundError as exc:
        sys.exit(f"{exc} — rode com ~/openarm_ws/.venv312/bin/python -m "
                 "openarm_data_collection.policy_runner ... (ou pelo launch, que "
                 "já usa o interpretador certo via `prefix`)")

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)

    # `ros2 launch ... checkpoint:=~/x` NÃO expande o ~ (não está no início da
    # palavra). Sem isto o lerobot não acha o diretório, assume repo do Hub e
    # estoura um HFValidationError que não diz nada sobre o problema real.
    expandido = os.path.abspath(os.path.expanduser(ckpt))
    if os.path.isdir(expandido):
        ckpt = expandido

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
        # postprocessor fica em CPU de propósito: é ele que traz a ação de volta
        # pra host. Forçando "cuda" aqui, predict_action devolve tensor CUDA.
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post, torch


class PolicyRunner(Node):
    def __init__(self, args):
        super().__init__("policy_runner")
        self.args = args
        # Slots que a POLICY controla, montados pela MESMA função do recorder
        # (joint_layout.build_slots): a ordem do vetor de estado/ação não tem como
        # divergir entre gravar e executar. Num checkpoint unimanual (ex.: 8 dims, só
        # o braço direito) são 8 posições indexadas de 0 a 7; `pidx` traduz junta ROS
        # -> índice no vetor da policy. O braço fora de `--arm` nem recebe comando
        # (não é "segurado", simplesmente não é publicado).
        self.arm = check_arm(args.arm)
        self.active_slots = build_slots(arm=self.arm)
        self.active_sides = list(sides_for_arm(self.arm))
        self.n_dims = len(self.active_slots)
        self.pidx = {s.ros_joint: s.index for s in self.active_slots}
        self.device = args.device

        self.get_logger().info(f"carregando {args.checkpoint} em {self.device}...")
        self.policy, self.pre, self.post, self.torch = _load_policy(
            args.checkpoint, self.device, args.torch_threads)
        # importado uma vez só: isto roda a cada tick dentro do callback de sync.
        from lerobot.common.control_utils import predict_action
        self._predict = predict_action

        self._setup_rename_map()
        self._check_action_dims()
        self._setup_cameras()
        self._setup_action_horizon()
        self._setup_temporal_ensembling()
        self._warmup()

        # ---- estado ----------------------------------------------------
        self.armed = bool(args.auto_arm)
        self.js: dict[str, float] | None = None
        self.last_cmd: dict[str, float] | None = None   # último alvo publicado
        self.last_obs_t = 0.0
        self.ramp_until = 0.0
        self.ramp_from: dict[str, float] | None = None
        self.n_actions = 0
        self._t_log = 0.0
        self._n_log = 0

        # ---- publishers -------------------------------------------------
        # RELIABLE serve tanto sub RELIABLE quanto BEST_EFFORT. O sub do
        # ForwardCommandController é BEST_EFFORT; publicar RELIABLE conecta nos dois.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.arm_pub = {
            "left": self.create_publisher(Float64MultiArray, args.left_arm_topic, qos),
            "right": self.create_publisher(Float64MultiArray, args.right_arm_topic, qos),
        }
        self.grip_pub = {
            "left": self.create_publisher(JointTrajectory, args.left_gripper_topic, qos),
            "right": self.create_publisher(JointTrajectory, args.right_gripper_topic, qos),
        }
        self.create_subscription(Bool, "/policy_runner/enable", self._on_enable, 1)

        # ---- observação: MESMO sync do episode_recorder ------------------
        img_type = CompressedImage if args.cameras_compressed else Image
        # Só as câmeras que a policy consome: o synchronizer só dispara quando
        # TODOS os tópicos assinados têm mensagem, então assinar uma câmera inútil
        # (ex.: pulso esquerdo numa policy do braço direito) faz o laço inteiro
        # parar se ela cair.
        subs = [Subscriber(self, JointState, "/joint_states")]
        subs += [Subscriber(self, img_type, topic) for _, topic in self.cams]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=30, slop=args.sync_slop)
        self.sync.registerCallback(self._on_obs)

        self.get_logger().info(
            f"pronto | {'DRY-RUN (não publica)' if args.dry_run else 'AO VIVO'} | "
            f"{'ARMADO' if self.armed else 'DESARMADO — publique True em /policy_runner/enable'} | "
            f"arm={self.arm} ({self.n_dims} dims) | cameras={[k.split('.')[-1] for k, _ in self.cams]} | "
            f"max_step={args.max_step} rad/tick | freeze_left={args.freeze_left}")

    # ------------------------------------------------------------------
    # configuração dependente da policy
    # ------------------------------------------------------------------
    def _setup_rename_map(self) -> None:
        """Lê o rename de features gravado no preprocessor do checkpoint.

        O rename roda DENTRO do preprocessor (é o passo 0 na SmolVLA), então
        temos que alimentar as chaves de ORIGEM. Filtrar contra
        `config.input_features` — que já tem os nomes de destino — descartava
        todas as imagens e a policy estourava "All image features are missing".
        """
        self.rename_map: dict[str, str] = {}
        for step in getattr(self.pre, "steps", []):
            rm = getattr(step, "rename_map", None)
            if rm:
                self.rename_map = dict(rm)
                self.get_logger().info(f"rename do checkpoint: {self.rename_map}")
                break
        imagens = sorted(set(self.policy.config.input_features) - {"observation.state"})
        self.get_logger().info(f"policy '{self.policy.config.type}' espera as imagens: {imagens}")

    def _check_action_dims(self) -> None:
        """Falha ALTO na partida se a ação da policy não bate com `--arm`.

        Antes isto só aparecia como "tick ignorado" repetido a 30 Hz. O tamanho do
        STATE em config.json não serve de checagem: checkpoints do crislmfroes e do
        gdorneles declaram state [6] (metadado morto), mas a ação [8]/[16] é real.
        """
        feat = self.policy.config.output_features.get("action")
        n_pol = feat.shape[0] if feat is not None else None
        if n_pol != self.n_dims:
            arm_sugerido = {16: "both", 8: "right (ou left)"}.get(n_pol, "?")
            sys.exit(f"a policy produz {n_pol} dims de ação, mas --arm={self.arm} usa "
                     f"{self.n_dims}. Use --arm {arm_sugerido}.")

    def _setup_cameras(self) -> None:
        """Descobre quais das 3 câmeras do robô a policy realmente consome.

        Vale a chave de ORIGEM (cam_chest...) ou o destino do rename (camera1...)
        estar em input_features. A mesma lista alimenta assinatura, warmup e
        inferência, então não tem como o warmup rodar com um conjunto de imagens
        diferente do real.
        """
        todas = [("observation.images.cam_chest", self.args.cam_chest_topic),
                 ("observation.images.left_cam_wrist", self.args.left_wrist_topic),
                 ("observation.images.right_cam_wrist", self.args.right_wrist_topic)]
        esperadas = set(self.policy.config.input_features)
        self.cams = [(k, t) for k, t in todas if self.rename_map.get(k, k) in esperadas]
        if not self.cams:
            sys.exit(f"nenhuma câmera do robô bate com as imagens da policy: {sorted(esperadas)}")

    def _setup_action_horizon(self) -> None:
        """Horizonte recuante: quantas ações do chunk executar antes de re-planejar.

        O default do checkpoint (50 na SmolVLA, 100 no ACT) é tempo em malha
        aberta: a 30 Hz são 1,67 s e 3,3 s sem olhar a observação. Reduzir corrige
        mais vezes, mas cada recálculo custa ~170 ms (SmolVLA no zap) e vira
        engasgo. Medido no zap (RTX 5070 Ti Laptop), taxa sustentável:
        n=1 -> 6.1 Hz | n=5 -> 25.5 Hz | n=10 -> 44.3 Hz | n=50 -> 110 Hz.
        Abaixo de 30 Hz a base de tempo deixa de bater com a da gravação.
        """
        if not self.args.n_action_steps:
            return
        antes = self.policy.config.n_action_steps
        self.policy.config.n_action_steps = self.args.n_action_steps
        self.policy.reset()
        self.get_logger().info(
            f"n_action_steps {antes} -> {self.args.n_action_steps} "
            f"(re-planeja a cada {self.args.n_action_steps / 30:.2f}s @30Hz)")

    def _setup_temporal_ensembling(self) -> None:
        """Só o ACT implementa temporal ensembling; nas outras policies é ignorado."""
        pedido = self.args.temporal_ensemble
        is_act = self.policy.config.type == "act"
        if pedido is not None and not is_act:
            self.get_logger().warn(
                f"--temporal-ensemble IGNORADO: só vale para ACT, e esta policy é "
                f"'{self.policy.config.type}' (ela re-prediz a cada "
                f"{self.policy.config.n_action_steps} ações — use --n-action-steps).")
        elif pedido is not None:
            _enable_temporal_ensembling(self.policy, pedido, self.get_logger())
        elif is_act:
            self.get_logger().warn(
                f"temporal ensembling DESLIGADO — o ACT vai executar "
                f"{self.policy.config.n_action_steps} ações em malha aberta "
                f"({self.policy.config.n_action_steps / 30:.1f}s sem corrigir). "
                "Considere --temporal-ensemble 0.01")

    def _warmup(self) -> None:
        """Queima o custo da primeira inferência (init lazy de CUDA).

        Medido: 392 ms no ACT e 544 ms na SmolVLA, contra ~3 ms nos ticks
        seguintes. Sem isto, essa trava cairia bem no instante em que você arma.
        """
        obs = {"observation.state": np.zeros(self.n_dims, dtype=np.float32)}
        for key, _ in self.cams:   # nomes de ORIGEM, que é o que o preprocessor espera
            obs[key] = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
        t0 = time.time()
        self._predict(obs, self.policy, self.torch.device(self.device), self.pre, self.post,
                      use_amp=False, task=self.args.task, robot_type=self.args.robot_type)
        self.policy.reset()
        self.get_logger().info(f"warmup: {(time.time() - t0) * 1000:.0f} ms (descartado)")

    # ------------------------------------------------------------------
    # laço de controle
    # ------------------------------------------------------------------
    def _on_enable(self, msg: Bool) -> None:
        if msg.data and not self.armed:
            self.armed = True
            self.ramp_from = dict(self.js) if self.js else None
            self.ramp_until = time.time() + self.args.ramp_seconds
            self.get_logger().warn(f"⚠️  ARMADO — rampa de {self.args.ramp_seconds}s")
        elif not msg.data and self.armed:
            self.armed = False
            self.hold()
            self.get_logger().warn("DESARMADO — segurando pose atual")

    def _state_vector(self, js: JointState) -> np.ndarray:
        idx = {n: i for i, n in enumerate(js.name)}
        out = np.zeros(self.n_dims, dtype=np.float32)
        for i, s in enumerate(self.active_slots):
            if s.ros_joint in idx:
                out[i] = js.position[idx[s.ros_joint]]
        return out

    def _on_obs(self, js, *imgs) -> None:
        self.js = {n: p for n, p in zip(js.name, js.position)}
        self.last_obs_t = time.time()
        state = self._state_vector(js)

        obs = {"observation.state": state}
        for (key, _), msg in zip(self.cams, imgs):
            obs[key] = to_dataset_frame(msg, self.args.image_height, self.args.image_width)

        action = self._predict(
            obs, self.policy, self.torch.device(self.device), self.pre, self.post,
            use_amp=False, task=self.args.task, robot_type=self.args.robot_type,
        )
        if hasattr(action, "detach"):
            action = action.detach().to("cpu")
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.size != self.n_dims:
            self.get_logger().error(
                f"policy devolveu {act.size} dims, esperado {self.n_dims} — tick ignorado")
            return

        self.n_actions += 1
        self.dispatch(act, state)

    def _target_per_joint(self, act: np.ndarray, state: np.ndarray):
        """policy -> alvo por junta ROS, com clamp de limite e rate-limit.

        Devolve (alvos, cortadas_por_urdf, cortadas_por_rate) — os dois conjuntos
        separados porque o diagnóstico depende de saber QUAL limite agiu.
        """
        base = self.last_cmd if self.last_cmd is not None else {
            s.ros_joint: float(state[i]) for i, s in enumerate(self.active_slots)}
        out: dict[str, float] = {}
        por_urdf: list[str] = []
        por_rate: list[str] = []
        for i, s in enumerate(self.active_slots):
            if self.args.freeze_left and s.side == "left":
                out[s.ros_joint] = base.get(s.ros_joint, float(state[i]))
                continue
            cru = float(act[i])
            lo, hi = URDF_LIMITS_RAD.get(s.ros_joint, (-np.inf, np.inf))
            tgt = min(max(cru, lo), hi)                        # 1) limite do URDF
            if tgt != cru:
                por_urdf.append(s.ros_joint)
            prev = base.get(s.ros_joint, float(state[i]))
            step = self.args.gripper_max_step if s.is_gripper else self.args.max_step
            limitado = prev + float(np.clip(tgt - prev, -step, step))   # 2) rate limit
            if abs(limitado - tgt) > 1e-9:
                por_rate.append(s.ros_joint)
            out[s.ros_joint] = limitado
        return out, por_urdf, por_rate

    def dispatch(self, act: np.ndarray, state: np.ndarray) -> None:
        target, por_urdf, por_rate = self._target_per_joint(act, state)
        now = time.time()

        if not self.armed or self.args.dry_run:
            self._log_parado(act, state, por_urdf, now)
            return

        # rampa: no primeiro instante após armar, interpola do medido até o alvo
        if self.ramp_from and now < self.ramp_until:
            a = 1.0 - (self.ramp_until - now) / max(self.args.ramp_seconds, 1e-3)
            a = 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0, 1))   # ease cosseno
            target = {j: self.ramp_from.get(j, v) * (1 - a) + v * a for j, v in target.items()}
        elif self.ramp_from and now >= self.ramp_until:
            self.ramp_from = None
            self.get_logger().info("rampa concluída — policy no controle")

        self._log_armado(act, target, por_urdf, por_rate, now)
        self.publish(target)
        self.last_cmd = target

    # ------------------------------------------------------------------
    # diagnóstico
    # ------------------------------------------------------------------
    def _deve_logar(self, now: float) -> float | None:
        """True (devolvendo o Hz medido) a cada LOG_PERIOD_S de tempo real.

        Em tempo e não a cada N ticks: com --n-action-steps baixo o laço cai para
        ~6 Hz e um log "a cada 30 ticks" viraria um a cada 5 segundos.
        """
        if self._t_log and now - self._t_log < LOG_PERIOD_S:
            return None
        hz = (self.n_actions - self._n_log) / (now - self._t_log) if self._t_log else float("nan")
        self._t_log, self._n_log = now, self.n_actions
        return hz

    def _log_parado(self, act, state, por_urdf, now: float) -> None:
        """Log de dry-run/desarmado: olha a ação CRUA da policy, nos DOIS braços.

        Tem que ser a crua: o `target` já passou pelo rate-limit, então o delta
        dele nunca passaria de --max-step e o log pareceria saudável mesmo com a
        policy pedindo absurdo. E tem que ser bilateral — com freeze_left ligado
        o esquerdo vira 0 no target de qualquer jeito, mas o CRU mostra o que a
        policy pediria se o freeze não estivesse lá (útil pra saber se compensa
        desligar --freeze-left numa tarefa bimanual).
        """
        if self._deve_logar(now) is None:
            return
        partes = []
        for lado, rotulo in (("right", "dir"), ("left", "esq")):
            juntas = [s for s in self.active_slots if s.side == lado]
            if not juntas:
                continue   # braço fora de --arm
            pior = max(juntas, key=lambda s: abs(act[self.pidx[s.ros_joint]] - state[self.pidx[s.ros_joint]]))
            delta = float(act[self.pidx[pior.ros_joint]] - state[self.pidx[pior.ros_joint]])
            partes.append(f"{rotulo} {pior.ros_joint} {delta:+.4f}")
        msg = (f"[{'seco' if self.args.dry_run else 'desarmado'}] ação #{self.n_actions} "
               f"| maior delta CRU: {' | '.join(partes)} rad "
               f"(seria limitado a {self.args.max_step:.3f}/tick)")
        if por_urdf:
            msg += f" | ⚠️ FORA DO LIMITE URDF: {por_urdf}"
        self.get_logger().info(msg)

    def _log_armado(self, act, target, por_urdf, por_rate, now: float) -> None:
        """Log durante operação armada.

        Sem isto o nó fica mudo exatamente quando importa, e não dá pra distinguir
        "a policy não pede movimento" de "o limitador está cortando o movimento".
        Os dois cortes aparecem separados de propósito: atribuir ao max_step um
        corte que veio do limite do URDF manda você mexer no parâmetro errado.
        """
        hz = self._deve_logar(now)
        if hz is None or not self.last_cmd:
            return
        # Bilateral: com freeze_left ligado o lado esquerdo aparece ~0 aplicado
        # (confirma visualmente que o freeze está segurando), sem custar uma
        # condição — o cálculo é o mesmo dos dois lados.
        partes = []
        for lado, rotulo in (("right", "dir"), ("left", "esq")):
            bracos = [s for s in self.active_slots if s.side == lado and not s.is_gripper]
            if not bracos:
                continue   # braço fora de --arm
            pedido = max(abs(act[self.pidx[s.ros_joint]] - self.last_cmd[s.ros_joint]) for s in bracos)
            aplicado = max(abs(target[s.ros_joint] - self.last_cmd[s.ros_joint]) for s in bracos)
            garra = next(s for s in self.active_slots if s.side == lado and s.is_gripper)
            partes.append(f"{rotulo} pediu {pedido:.4f} aplicou {aplicado:.4f} | garra {rotulo} {target[garra.ros_joint]:.3f}")
        msg = f"#{self.n_actions} {hz:4.1f}Hz | " + " || ".join(partes)
        if por_rate:
            msg += f" | CORTADO por max_step: {len(por_rate)} junta(s)"
        if por_urdf:
            msg += f" | ⚠️ CLAMP do URDF: {por_urdf}"
        self.get_logger().info(msg)

    # ------------------------------------------------------------------
    # saída
    # ------------------------------------------------------------------
    def publish(self, target: dict[str, float]) -> None:
        for side in self.active_sides:   # o braço fora de --arm não recebe nada
            braco = [s for s in self.active_slots if s.side == side and not s.is_gripper]
            m = Float64MultiArray()
            m.data = [float(target[s.ros_joint]) for s in braco]
            self.arm_pub[side].publish(m)

            garra = [s for s in self.active_slots if s.side == side and s.is_gripper]
            t = JointTrajectory()
            t.header.stamp = self.get_clock().now().to_msg()
            t.joint_names = [s.ros_joint for s in garra]
            pt = JointTrajectoryPoint()
            pt.positions = [float(target[s.ros_joint]) for s in garra]
            pt.time_from_start = Duration(sec=0, nanosec=int(0.1 * 1e9))
            t.points = [pt]
            self.grip_pub[side].publish(t)

    def hold(self) -> bool:
        """Segura a pose medida agora (desarme e saída). True se conseguiu publicar.

        No Ctrl-C isto costuma FALHAR: o SIGINT já invalidou o contexto do rclpy
        antes de chegarmos aqui, e o publish estoura RCLError. Não é grave — o
        ForwardCommandController segura o último comando sozinho quando o stream
        para —, mas é melhor avisar do que fingir que seguramos a pose.
        """
        if self.js is None:
            return False
        target = {s.ros_joint: float(self.js.get(s.ros_joint, 0.0)) for s in self.active_slots}
        try:
            self.publish(target)
        except Exception as exc:
            self.get_logger().warn(
                f"hold não publicou ({type(exc).__name__}) — o controller mantém o "
                "último comando, o braço não se move sozinho")
            return False
        self.last_cmd = target
        return True

    def watchdog(self) -> None:
        if not self.armed or self.args.dry_run:
            return
        parado = time.time() - self.last_obs_t
        if self.last_obs_t and parado > self.args.obs_timeout:
            self.get_logger().error(
                f"observação parada há {parado:.2f}s (> {self.args.obs_timeout}) — DESARMANDO")
            self.armed = False
            self.hold()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Roda uma policy do LeRobot no OpenArm via ros2_control.")
    ap.add_argument("--checkpoint", required=True,
                    help="pasta pretrained_model/ do checkpoint, ou repo id do HuggingFace")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-threads", type=int, default=2,
                    help="threads de CPU do torch. O default do PyTorch é 1 por core "
                         "(24 no zap), que sufoca os drivers de câmera e o ros2_control "
                         "na mesma máquina.")
    ap.add_argument("--task", default="put the lego brick in the box",
                    help="string da tarefa. Tem que ser a EXATA do dataset de treino: "
                         "policies com VLM (SmolVLA, pi0) tokenizam e condicionam nela; "
                         "o ACT ignora por completo.")
    ap.add_argument("--robot-type", default="bi_openarm_follower")
    ap.add_argument("--dry-run", action="store_true",
                    help="roda a policy mas não publica nos controllers")
    ap.add_argument("--auto-arm", action="store_true", help="já começa armado (cuidado)")

    ap.add_argument("--arm", choices=ARMS, default="both",
                    help="qual(is) braço(s) a policy controla. both = 16 dims; right/left = "
                         "8 dims (checkpoint unimanual). O outro braço não recebe comando "
                         "nenhum e as câmeras que a policy não usa nem são assinadas.")

    seg = ap.add_argument_group("segurança")
    # Defaults calibrados contra o dataset (|action[t]-action[t-1]|, braço direito):
    #   média 0.0098 | p95 0.0378 | p99 0.0623 | max 0.3633 rad/tick
    # 0.03 corta ~10% dos ticks mais rápidos (movimento mais suave); 0.04 ≈ p95 passa
    # a demo quase intacta; o max de 0.36 (10.9 rad/s) é glitch de teleop, e é
    # justamente o que se quer cortar.
    # ATENÇÃO: o limite é relativo ao ÚLTIMO COMANDO, não ao estado medido — existe
    # um offset permanente de tracking (|action-state| média 0.069, p95 0.23 rad),
    # então limitar contra o medido brigaria com a operação normal.
    seg.add_argument("--max-step", type=float, default=0.03,
                     help="rad que cada junta do braço pode andar por tick "
                          "(a 30Hz, 0.03 = 0.9 rad/s)")
    # garra no dataset: média 0.00043 | p95 0.00327 | max 0.0226 m/tick
    seg.add_argument("--gripper-max-step", type=float, default=0.004,
                     help="metros por tick da garra prismática (0..0.044); ≈ p95 do dataset")
    seg.add_argument("--ramp-seconds", type=float, default=3.0,
                     help="interpolação suave do estado medido até a 1ª ação, ao armar")
    seg.add_argument("--obs-timeout", type=float, default=0.5,
                     help="sem observação nova por este tempo, desarma")
    seg.add_argument("--freeze-left", dest="freeze_left", action="store_true", default=False,
                     help="ignora as 8 dims do braço esquerdo (padrão: DESLIGADO — "
                          "ligue só se souber que a tarefa não usa o braço esquerdo)")
    seg.add_argument("--no-freeze-left", dest="freeze_left", action="store_false")

    inf = ap.add_argument_group("inferência")
    inf.add_argument("--n-action-steps", type=int, default=None, metavar="N",
                     help="ações do chunk executadas antes de re-planejar (horizonte "
                          "recuante). Vazio = o do checkpoint. Taxa sustentável medida "
                          "no zap: 1->6Hz, 5->25Hz, 10->44Hz, 50->110Hz.")
    inf.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                     help="temporal ensembling do ACT (recomendado 0.01). Re-prediz a "
                          "cada tick em vez de executar o chunk às cegas. Ignorado se a "
                          "policy não for ACT.")

    obs = ap.add_argument_group("observação (tem que bater com o recording_params.yaml da gravação)")
    obs.add_argument("--image-height", type=int, default=480)
    obs.add_argument("--image-width", type=int, default=640)
    obs.add_argument("--sync-slop", type=float, default=0.033)
    obs.add_argument("--cameras-compressed", action="store_true")
    obs.add_argument("--cam-chest-topic", default="/cam_chest/color/image_raw")
    obs.add_argument("--left-wrist-topic", default="/left_wrist/image_raw")
    obs.add_argument("--right-wrist-topic", default="/right_wrist/image_raw")

    cmd = ap.add_argument_group("comandos")
    cmd.add_argument("--left-arm-topic", default="/left_forward_position_controller/commands")
    cmd.add_argument("--right-arm-topic", default="/right_forward_position_controller/commands")
    cmd.add_argument("--left-gripper-topic", default="/left_gripper_controller/joint_trajectory")
    cmd.add_argument("--right-gripper-topic", default="/right_gripper_controller/joint_trajectory")
    return ap


def main(argv=None):
    args, _ = build_parser().parse_known_args(argv)

    rclpy.init()
    node = PolicyRunner(args)
    node.create_timer(0.1, node.watchdog)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # SIGINT ou SIGTERM
        node.get_logger().warn("encerrando (sinal) — desarmando")
        node.armed = False
        node.hold()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
