"""학습된 모델을 달리게 해서 스텝별 궤적을 돌려준다 (대시보드의 '주행 시험'용)."""
import json
import os

from .env import FPS, MiniRacerEnv, auto_max_steps, load_action_space
from .ppo import PPOAgent
from .track import Track, resolve_track


def play_trajectories(run_dir, tracks_dir, model="best", track=None, noise=0.0, laps=3, seed=0):
    info = json.load(open(os.path.join(run_dir, "run_info.json"), encoding="utf-8"))
    model_path = os.path.join(run_dir, f"model_{model}.npz")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: model_{model}.npz (아직 저장되지 않았을 수 있습니다)")

    try:
        track_file = resolve_track(track or info.get("track_file") or info["track"], tracks_dir)
    except FileNotFoundError:
        track_file = resolve_track(info["track"], tracks_dir)
    tr = Track(track_file)

    meta_path = os.path.join(run_dir, "model_metadata.json")
    reward_path = os.path.join(run_dir, "reward_function.py")
    env = MiniRacerEnv(tr, reward_path, meta_path, auto_max_steps(tr, load_action_space(meta_path)), noise, seed)
    agent = PPOAgent.load(model_path)

    episodes = []
    for k in range(laps):
        start = int(tr.n * k / max(1, laps)) if k else 0
        obs = env.reset(start)
        xs, ys, sp, st, pr = [env.x], [env.y], [0.0], [0.0], [0.0]
        total = 0.0
        while True:
            obs, r, done, inf = env.step(agent.act(obs, deterministic=True))
            total += r
            xs.append(env.x)
            ys.append(env.y)
            sp.append(env.speed)
            st.append(env.steer)
            pr.append(min(100.0, env.progress))
            if done:
                break
        ok = inf["status"] == "lap_complete"
        episodes.append({
            "start_idx": start, "status": inf["status"], "progress": round(min(100.0, env.progress), 1),
            "lap_time": round(env.steps / FPS, 2) if ok else None, "total_reward": round(total, 1),
            "x": [round(v, 3) for v in xs], "y": [round(v, 3) for v in ys],
            "speed": [round(v, 2) for v in sp], "steer": [round(v, 1) for v in st], "progress_t": [round(v, 1) for v in pr],
        })
    return {"track": tr.name, "trained_track": info["track"], "noise": noise, "episodes": episodes}
