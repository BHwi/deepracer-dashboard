"""학습된 모델로 주행만 해 본다 (탐험 없이 가장 확률 높은 행동만 선택).

예)
  python play.py --run my_first                          # runs/my_first/model_best.npz, 학습한 트랙
  python play.py --run my_first --laps 5
  python play.py --run my_first --track Oval_track       # 다른 트랙에서 주행 (일반화 실험)
  python play.py --run my_first --noise 1.0              # 외란을 주면 얼마나 무너지나 (Sim-to-Real 실험)
  python play.py --run my_first --model final            # model_final.npz 사용
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import sys

import numpy as np

from miniracer.env import MiniRacerEnv, RewardFunctionError, auto_max_steps, load_action_space, FPS
from miniracer.ppo import PPOAgent
from miniracer.track import Track, resolve_track

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.join(HERE, "tracks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="runs/ 아래 실험 이름 (또는 폴더 경로)")
    p.add_argument("--model", default="best", help="best | final | npz 파일 경로")
    p.add_argument("--track", default=None, help="다른 트랙에서 달리기 (기본: 학습한 트랙)")
    p.add_argument("--laps", type=int, default=3, help="시도 횟수 (출발 위치를 조금씩 바꿔가며)")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-rays", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join(HERE, "runs", args.run)
    if not os.path.isdir(run_dir):
        sys.exit(f"실험 폴더가 없습니다: {run_dir}")
    info = json.load(open(os.path.join(run_dir, "run_info.json"), encoding="utf-8"))
    model_path = args.model if args.model.endswith(".npz") else os.path.join(run_dir, f"model_{args.model}.npz")

    track_name = args.track or info.get("track_file") or info["track"]
    try:
        track_file = resolve_track(track_name, TRACKS_DIR)
    except FileNotFoundError:
        track_file = resolve_track(info["track"], TRACKS_DIR)
    track = Track(track_file)

    meta_path = os.path.join(run_dir, "model_metadata.json")
    reward_path = os.path.join(run_dir, "reward_function.py")
    actions = load_action_space(meta_path)
    try:
        env = MiniRacerEnv(track, reward_path, meta_path, auto_max_steps(track, actions), args.noise, args.seed)
    except RewardFunctionError as e:
        sys.exit(f"[보상함수 오류] {e}")
    agent = PPOAgent.load(model_path)

    viewer = None
    if not args.no_render:
        from miniracer.viewer import Viewer
        viewer = Viewer(env, title=f"MiniRacer play - {os.path.basename(run_dir)}", show_rays=not args.no_rays)

    print(f"모델 {model_path}\n트랙 {track.name} (학습 트랙: {info['track']}) | 외란 {args.noise}")
    done_laps, times = 0, []
    for k in range(args.laps):
        start = int(track.n * k / max(1, args.laps)) if k else 0
        obs = env.reset(start)
        total = 0.0
        if viewer:
            viewer.begin_episode()
        while True:
            obs, r, done, info_s = env.step(agent.act(obs, deterministic=True))
            total += r
            if viewer:
                viewer.update_step(r, f"try {k + 1}/{args.laps}")
            if done:
                break
        ok = info_s["status"] == "lap_complete"
        lap = f"{env.steps / FPS:.2f}s" if ok else "-"
        print(f"시도 {k + 1}: {info_s['status']:12s} 진행률 {min(100.0, env.progress):5.1f}%  랩타임 {lap}  총보상 {total:.1f}")
        done_laps += ok
        if ok:
            times.append(env.steps / FPS)
    print(f"완주 {done_laps}/{args.laps}" + (f" | 평균 랩타임 {np.mean(times):.2f}s" if times else ""))
    if viewer:
        import matplotlib.pyplot as plt
        print("창을 닫으면 종료됩니다.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
