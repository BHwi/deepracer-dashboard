"""PPO (clipped) - NumPy 구현.

DeepRacer 의 기본 알고리즘과 같은 계열(clipped PPO, 이산 행동, categorical 탐험)이다.
차이점: 신경망이 'CNN(카메라 이미지)'가 아니라 '작은 MLP(레이 센서)'이고, 프레임워크 없이 NumPy 로만 짰다.

하이퍼파라미터 이름과 기본값은 DRfC 저장소의 defaults/hyperparameters.json 을 읽고 맞췄다.
  batch_size              경사하강 미니배치 크기
  num_epochs              같은 데이터로 반복해서 학습하는 횟수
  lr                      학습률
  beta_entropy            엔트로피 보너스 (탐험 유지)
  discount_factor         할인율
  loss_type               value 손실 (huber / mean squared error)
  num_episodes_between_training  정책을 한 번 갱신하기 전에 모으는 에피소드 수 (= 1 iteration)
"""
import json

import numpy as np

DEFAULT_HP = {
    # --- DRfC 저장소의 defaults/hyperparameters.json 과 같은 이름·기본값 (term_cond_avg_score 만 custom_files 에서 크게 설정) ---
    "batch_size": 64,
    "beta_entropy": 0.01,
    "discount_factor": 0.99,
    "loss_type": "huber",
    "lr": 0.0003,
    "num_episodes_between_training": 20,
    "num_epochs": 5,
    "term_cond_avg_score": 350.0,
    "term_cond_max_episodes": 1000,
    "exploration_type": "categorical",   # (MiniRacer 는 항상 categorical)
    "e_greedy_value": 0.05,              # (사용 안 함, 호환용)
    "epsilon_steps": 10000,              # (사용 안 함, 호환용)
    "stack_size": 1,                     # (사용 안 함, 호환용)
    # --- MiniRacer 에서 추가한 값 ---
    "clip_epsilon": 0.2,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "min_eval_trials": 3,
    "round_robin_advance_dist": 0.05,
    "max_steps_per_episode": "auto",
}


def load_hyperparameters(path):
    hp = dict(DEFAULT_HP)
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        unknown = [k for k in user if k not in DEFAULT_HP]
        if unknown:
            print(f"[경고] hyperparameters.json 에서 알 수 없는 키를 무시합니다: {unknown}")
        hp.update({k: v for k, v in user.items() if k in DEFAULT_HP})
    return hp


# ----------------------------------------------------------------------------- 신경망
class MLP:
    """tanh 은닉층 MLP (수동 역전파)."""

    def __init__(self, sizes, rng, last_gain=1.0):
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = last_gain if i == len(sizes) - 2 else 1.0
            self.W.append(rng.standard_normal((a, c)) * gain / np.sqrt(a))
            self.b.append(np.zeros(c))
        self.L = len(self.W)

    def params(self):
        return self.W + self.b

    def forward(self, x):
        hs = [x]
        for i in range(self.L):
            z = hs[-1] @ self.W[i] + self.b[i]
            hs.append(np.tanh(z) if i < self.L - 1 else z)
        return hs[-1], hs

    def backward(self, hs, dout):
        gW, gb = [None] * self.L, [None] * self.L
        g = dout
        for i in reversed(range(self.L)):
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - hs[i] ** 2)
        return gW + gb


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_by_global_norm(grads, max_norm):
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        grads = [g * scale for g in grads]
    return grads


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class RunningRewardScale:
    """할인 누적 보상의 표준편차로 보상 크기를 맞춘다 (내부용: 학습 안정화).
    보상함수의 절대 크기에 상관없이 학습이 되도록 하기 위한 장치."""

    def __init__(self, gamma):
        self.gamma, self.ret = gamma, 0.0
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, rewards):
        self.ret = 0.0
        for r in rewards:
            self.ret = self.gamma * self.ret + r
            self.count += 1
            d = self.ret - self.mean
            self.mean += d / self.count
            self.m2 += d * (self.ret - self.mean)

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.m2 / self.count) + 1e-8)


# ----------------------------------------------------------------------------- PPO
class PPOAgent:
    def __init__(self, obs_dim, n_actions, hp=None, seed=0, hidden=64):
        self.hp = dict(DEFAULT_HP if hp is None else hp)
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, hidden, hidden, n_actions], self.rng, last_gain=0.01)
        self.critic = MLP([obs_dim, hidden, hidden, 1], self.rng, last_gain=1.0)
        self.opt_a = Adam(self.actor.params(), self.hp["lr"])
        self.opt_c = Adam(self.critic.params(), self.hp["lr"])
        self.rscale = RunningRewardScale(self.hp["discount_factor"])

    # ------------------------------------------------------------------ 행동
    def probs(self, obs):
        logits, _ = self.actor.forward(obs[None, :])
        return softmax(logits[0])

    def act(self, obs, deterministic=False):
        p = self.probs(obs)
        if deterministic:
            return int(np.argmax(p))
        return int(min(np.searchsorted(np.cumsum(p), self.rng.random()), self.n_actions - 1))

    # ------------------------------------------------------------------ 학습
    def update(self, episodes):
        """episodes: dict 리스트. 각 dict = obs(T,d) act(T,) rew(T,) terminal(bool) last_obs(d,)
        (terminal=False 는 시간초과로 끊긴 에피소드 -> last_obs 의 가치로 부트스트랩)"""
        hp = self.hp
        gamma, lam = hp["discount_factor"], hp["gae_lambda"]

        for ep in episodes:
            self.rscale.update(ep["rew"])
        sd = self.rscale.std

        obs_all = np.concatenate([ep["obs"] for ep in episodes])
        v_all = self.critic.forward(obs_all)[0][:, 0]

        advs, rets, off = [], [], 0
        for ep in episodes:
            T = len(ep["rew"])
            r = np.asarray(ep["rew"]) / sd
            v = v_all[off:off + T]
            off += T
            v_last = 0.0 if ep["terminal"] else float(self.critic.forward(ep["last_obs"][None, :])[0][0, 0])
            adv = np.zeros(T)
            last = 0.0
            for t in reversed(range(T)):
                nv = v_last if t == T - 1 else v[t + 1]
                delta = r[t] + gamma * nv - v[t]
                last = delta + gamma * lam * last
                adv[t] = last
            advs.append(adv)
            rets.append(adv + v)
        adv_all = np.concatenate(advs)
        ret_all = np.concatenate(rets)
        act_all = np.concatenate([ep["act"] for ep in episodes]).astype(int)
        N = len(act_all)

        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        old_logits, _ = self.actor.forward(obs_all)
        old_p = softmax(old_logits)
        old_logp = np.log(old_p[np.arange(N), act_all] + 1e-12)

        clip, beta, bs = hp["clip_epsilon"], hp["beta_entropy"], int(hp["batch_size"])
        pol_losses, val_losses, clip_fracs = [], [], []
        for _ in range(int(hp["num_epochs"])):
            perm = self.rng.permutation(N)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                B = len(idx)
                x, a, A = obs_all[idx], act_all[idx], adv_all[idx]

                # ---- actor
                logits, hs = self.actor.forward(x)
                p = softmax(logits)
                logp_all = np.log(p + 1e-12)
                logp = logp_all[np.arange(B), a]
                ratio = np.exp(logp - old_logp[idx])
                active = ((A >= 0) & (ratio <= 1 + clip)) | ((A < 0) & (ratio >= 1 - clip))
                surr = np.minimum(ratio * A, np.clip(ratio, 1 - clip, 1 + clip) * A)
                H = -(p * logp_all).sum(axis=1)

                g_logp = np.where(active, -A * ratio, 0.0) / B
                onehot = np.zeros_like(p)
                onehot[np.arange(B), a] = 1.0
                d_logits = g_logp[:, None] * (onehot - p)
                d_logits += (beta / B) * p * (logp_all + H[:, None])   # 엔트로피 보너스의 기울기
                ga = clip_by_global_norm(self.actor.backward(hs, d_logits), hp["max_grad_norm"])
                self.opt_a.step(ga)

                pol_losses.append(float(-surr.mean()))
                clip_fracs.append(float((np.abs(ratio - 1) > clip).mean()))

                # ---- critic
                v, hc = self.critic.forward(x)
                err = v[:, 0] - ret_all[idx]
                if hp["loss_type"] == "huber":
                    d_v = np.clip(err, -1.0, 1.0) / B
                    vl = np.where(np.abs(err) <= 1, 0.5 * err ** 2, np.abs(err) - 0.5).mean()
                else:
                    d_v = err / B
                    vl = 0.5 * (err ** 2).mean()
                gc = clip_by_global_norm(self.critic.backward(hc, d_v[:, None]), hp["max_grad_norm"])
                self.opt_c.step(gc)
                val_losses.append(float(vl))

        new_p = softmax(self.actor.forward(obs_all)[0])
        kl = float((old_p * (np.log(old_p + 1e-12) - np.log(new_p + 1e-12))).sum(axis=1).mean())
        entropy = float(-(new_p * np.log(new_p + 1e-12)).sum(axis=1).mean())
        return {"policy_loss": float(np.mean(pol_losses)), "value_loss": float(np.mean(val_losses)),
                "entropy": entropy, "kl": kl, "clip_fraction": float(np.mean(clip_fracs)),
                "samples": N, "reward_scale": sd}

    # ------------------------------------------------------------------ 저장/불러오기
    def save(self, path):
        data = {"obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden}
        for i, w in enumerate(self.actor.W):
            data[f"aW{i}"], data[f"ab{i}"] = w, self.actor.b[i]
        for i, w in enumerate(self.critic.W):
            data[f"cW{i}"], data[f"cb{i}"] = w, self.critic.b[i]
        np.savez(path, **data)

    @classmethod
    def load(cls, path, hp=None):
        d = np.load(path)
        agent = cls(int(d["obs_dim"]), int(d["n_actions"]), hp, hidden=int(d["hidden"]))
        for i in range(agent.actor.L):
            agent.actor.W[i][:] = d[f"aW{i}"]
            agent.actor.b[i][:] = d[f"ab{i}"]
            agent.critic.W[i][:] = d[f"cW{i}"]
            agent.critic.b[i][:] = d[f"cb{i}"]
        return agent
