"""MiniRacer 학습.

DeepRacer 와 같은 흐름:
  1 iteration = (정책으로 num_episodes_between_training 개 에피소드 주행) -> 정책 갱신(PPO) -> 평가 주행

예)
  python train.py                                    # custom_files/ 의 3개 파일로 학습, 화면 표시
  python train.py --name my_first                    # 결과를 runs/my_first/ 에 저장
  python train.py --track Oval_track                 # 다른 트랙 (python train.py --list-tracks)
  python train.py --reward examples/rewards/02_speed_only.py --name speed_only
  python train.py --no-render                        # 화면 없이 빠르게
  python train.py --noise 1.0                        # 외란 (Sim-to-Real 실험)

결과: runs/<이름>/ 아래에
  reward_function.py, model_metadata.json, hyperparameters.json   (이 실험에 쓴 설정 사본)
  metrics.csv   iteration 별 지표 (보상, 진행률, 완주율, 랩타임, 엔트로피, KL, 손실)
  episodes.csv  에피소드별 결과
  trace_train.csv / trace_eval.csv   스텝별 기록 (DeepRacer 의 simulation trace 에 해당)
  model_best.npz / model_final.npz   (best = 평가 진행률이 가장 높았던 iteration)
  report.png    분석 리포트
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 작은 행렬 연산은 단일 스레드가 빠르다
    os.environ.setdefault(_k, "1")

import argparse
import csv
import json
import math
import shutil
import sys
import time
from datetime import datetime

import numpy as np

from miniracer.env import MiniRacerEnv, RewardFunctionError, auto_max_steps, load_action_space, FPS
from miniracer.ppo import PPOAgent, load_hyperparameters
from miniracer.track import Track, list_tracks, resolve_track

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.join(HERE, "tracks")
CUSTOM_DIR = os.path.join(HERE, "custom_files")

TRACE_COLS = ["iteration", "episode", "step", "x", "y", "heading", "steering_angle", "speed", "action", "reward",
              "progress", "all_wheels_on_track", "closest_waypoint", "distance_from_center", "episode_status"]
EP_COLS = ["phase", "iteration", "episode", "start_idx", "steps", "total_reward", "progress", "status", "lap_time_s"]
METRIC_COLS = ["iteration", "episodes", "steps", "train_reward", "train_progress", "train_lap_rate",
               "eval_reward", "eval_progress", "eval_lap_rate", "eval_lap_time_mean", "eval_lap_time_best",
               "entropy", "kl", "policy_loss", "value_loss", "clip_fraction", "elapsed_s"]


class _Tee:
    """화면 출력을 train.log 에도 같이 기록 (대시보드가 읽는다)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            try:
                st.write(text)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def write_status(run_dir, **kw):
    """status.json: 대시보드가 학습 상태를 읽는 파일 (임시 파일에 쓴 뒤 교체해서 읽는 쪽이 깨진 파일을 보지 않게 함)."""
    kw["updated"] = time.time()
    kw["pid"] = os.getpid()
    path = os.path.join(run_dir, "status.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(kw, f)
    os.replace(path + ".tmp", path)


def run_episode(env, agent, start_idx, deterministic, it, ep_no, trace_rows=None, viewer=None, label=""):
    """에피소드 1개 실행. returns (rollout dict, summary dict)."""
    obs = env.reset(start_idx)
    O, A, R = [], [], []
    if viewer:
        viewer.begin_episode()
    while True:
        a = agent.act(obs, deterministic)
        o2, r, done, info = env.step(a)
        O.append(obs)
        A.append(a)
        R.append(r)
        if trace_rows is not None:
            p = env.params
            trace_rows.append([it, ep_no, env.steps, round(env.x, 4), round(env.y, 4), round(env.heading, 2),
                               round(env.steer, 2), round(env.speed, 3), a, round(r, 4), round(env.progress, 3),
                               int(p["all_wheels_on_track"]), p["closest_waypoints"][0],
                               round(p["distance_from_center"], 4), info["status"] if done else "running"])
        if viewer:
            viewer.update_step(r, label)
        obs = o2
        if done:
            break
    status = info["status"]
    lap_time = env.steps / FPS if status == "lap_complete" else float("nan")
    roll = {"obs": np.array(O), "act": np.array(A), "rew": np.array(R), "terminal": status != "timeout", "last_obs": obs}
    summ = {"start_idx": start_idx, "steps": env.steps, "total_reward": float(sum(R)), "progress": float(min(100.0, env.progress)),
            "status": status, "lap_time": lap_time}
    return roll, summ


def make_run_dir(name, track_name):
    base = os.path.join(HERE, "runs")
    os.makedirs(base, exist_ok=True)
    name = name or f"{track_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    path, k = os.path.join(base, name), 2
    while os.path.exists(path):
        path = os.path.join(base, f"{name}_{k}")
        k += 1
    os.makedirs(path)
    return path


def main():
    p = argparse.ArgumentParser(description="MiniRacer PPO 학습")
    p.add_argument("--track", default="reInvent2019_track", help="트랙 이름 또는 npy 경로")
    p.add_argument("--name", default=None, help="실험 이름 (runs/<이름>/)")
    p.add_argument("--custom-files", default=CUSTOM_DIR, help="reward_function.py, model_metadata.json, hyperparameters.json 이 있는 폴더")
    p.add_argument("--reward", default=None, help="보상함수 파일만 따로 지정")
    p.add_argument("--hp", default=None, help="hyperparameters.json 만 따로 지정")
    p.add_argument("--metadata", default=None, help="model_metadata.json 만 따로 지정")
    p.add_argument("--render-every", type=int, default=5, help="N iteration 마다 첫 에피소드를 화면에 표시")
    p.add_argument("--no-render", action="store_true", help="화면 없이 실행")
    p.add_argument("--no-rays", action="store_true", help="화면에서 레이 센서 선 숨김")
    p.add_argument("--trace-every", type=int, default=3, help="N iteration 마다 학습 에피소드의 스텝별 기록 저장 (평가는 항상)")
    p.add_argument("--noise", type=float, default=0.0, help="외란 세기 (Sim-to-Real 실험)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=None, help="이 시간(분)이 지나면 종료")
    p.add_argument("--list-tracks", action="store_true")
    args = p.parse_args()

    if args.list_tracks:
        print("사용 가능한 트랙:", ", ".join(list_tracks(TRACKS_DIR)) or "(없음)")
        print("더 받으려면: python fetch_tracks.py")
        return

    reward_path = args.reward or os.path.join(args.custom_files, "reward_function.py")
    hp_path = args.hp or os.path.join(args.custom_files, "hyperparameters.json")
    meta_path = args.metadata or os.path.join(args.custom_files, "model_metadata.json")
    for f in (reward_path, hp_path, meta_path):
        if not os.path.isfile(f):
            sys.exit(f"파일을 찾을 수 없습니다: {f}")

    track_file = os.path.abspath(resolve_track(args.track, TRACKS_DIR))
    track = Track(track_file)
    hp = load_hyperparameters(hp_path)
    actions = load_action_space(meta_path)
    max_steps = auto_max_steps(track, actions) if hp["max_steps_per_episode"] == "auto" else int(hp["max_steps_per_episode"])

    try:
        env = MiniRacerEnv(track, reward_path, meta_path, max_steps, args.noise, args.seed)
        env_eval = MiniRacerEnv(track, reward_path, meta_path, max_steps, args.noise, args.seed + 1)
    except RewardFunctionError as e:
        sys.exit(f"[보상함수 오류] {e}")

    run_dir = make_run_dir(args.name, track.name)
    _logf = open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, _logf), _Tee(sys.stderr, _logf)
    shutil.copy(reward_path, os.path.join(run_dir, "reward_function.py"))
    shutil.copy(meta_path, os.path.join(run_dir, "model_metadata.json"))
    with open(os.path.join(run_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"track": track.name, "track_file": track_file, "noise": args.noise, "seed": args.seed, "max_steps": max_steps,
                   "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"실험 폴더: {run_dir}")
    print(f"트랙 {track.name} (길이 {track.length:.1f}m, 폭 {track.width:.2f}m) | 행동 {len(actions)}개 | "
          f"에피소드당 최대 {max_steps} 스텝 | 이터레이션당 {hp['num_episodes_between_training']} 에피소드")

    viewer = None
    if not args.no_render:
        from miniracer.viewer import Viewer
        viewer = Viewer(env, title=f"MiniRacer - {os.path.basename(run_dir)}", show_rays=not args.no_rays)

    agent = PPOAgent(env.obs_dim, env.n_actions, hp, seed=args.seed)
    f_tr = open(os.path.join(run_dir, "trace_train.csv"), "w", newline="", encoding="utf-8")
    f_ev = open(os.path.join(run_dir, "trace_eval.csv"), "w", newline="", encoding="utf-8")
    f_ep = open(os.path.join(run_dir, "episodes.csv"), "w", newline="", encoding="utf-8")
    f_me = open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8")
    w_tr, w_ev, w_ep, w_me = (csv.writer(f) for f in (f_tr, f_ev, f_ep, f_me))
    w_tr.writerow(TRACE_COLS)
    w_ev.writerow(TRACE_COLS)
    w_ep.writerow(EP_COLS)
    w_me.writerow(METRIC_COLS)

    curves = {k: [] for k in ("train_reward", "eval_reward", "train_progress", "eval_progress", "train_lap_rate", "entropy", "kl")}
    best_key, ep_no, total_steps, recent_rewards = None, 0, 0, []
    t0 = time.time()
    n_ep = int(hp["num_episodes_between_training"])
    n_eval = int(hp["min_eval_trials"])
    stop_reason = "term_cond_max_episodes"
    it = 0
    max_iters = int(math.ceil(hp["term_cond_max_episodes"] / n_ep))
    write_status(run_dir, state="running", iteration=0, max_iterations=max_iters, episodes=0, elapsed_s=0.0)
    try:
        while True:
            it += 1
            render_it = viewer is not None and args.render_every > 0 and it % args.render_every == 0
            trace_it = args.trace_every > 0 and it % args.trace_every == 0
            rolls, summs = [], []
            for k in range(n_ep):
                ep_no += 1
                # DeepRacer 훈련처럼 출발 위치를 에피소드마다 조금씩 옮긴다 (round robin)
                start = int(((ep_no - 1) * hp["round_robin_advance_dist"] % 1.0) * track.n)
                rows = [] if trace_it else None
                roll, s = run_episode(env, agent, start, False, it, ep_no, rows,
                                      viewer if (render_it and k == 0) else None, f"iter {it}  ep {ep_no}  (train)")
                if rows:
                    w_tr.writerows(rows)
                rolls.append(roll)
                summs.append(s)
                total_steps += s["steps"]
                recent_rewards.append(s["total_reward"])
                w_ep.writerow(["train", it, ep_no, s["start_idx"], s["steps"], round(s["total_reward"], 3),
                               round(s["progress"], 2), s["status"], "" if math.isnan(s["lap_time"]) else round(s["lap_time"], 3)])

            st = agent.update(rolls)

            ev = []
            for k in range(n_eval):
                rows = []
                ev_start = int(track.n * k / n_eval)          # 평가 출발 위치를 트랙에 고르게 분산
                _, s = run_episode(env_eval, agent, ev_start, True, it, ep_no + k + 1, rows)
                w_ev.writerows(rows)
                ev.append(s)
                w_ep.writerow(["eval", it, ep_no + k + 1, ev_start, s["steps"], round(s["total_reward"], 3), round(s["progress"], 2),
                               s["status"], "" if math.isnan(s["lap_time"]) else round(s["lap_time"], 3)])

            tr_rew = float(np.mean([s["total_reward"] for s in summs]))
            tr_prog = float(np.mean([s["progress"] for s in summs]))
            tr_lap = 100.0 * float(np.mean([s["status"] == "lap_complete" for s in summs]))
            ev_rew = float(np.mean([s["total_reward"] for s in ev]))
            ev_prog = float(np.mean([s["progress"] for s in ev]))
            ev_lap = 100.0 * float(np.mean([s["status"] == "lap_complete" for s in ev]))
            laps = [s["lap_time"] for s in ev if s["status"] == "lap_complete"]
            lap_mean = float(np.mean(laps)) if laps else float("nan")
            lap_best = float(np.min(laps)) if laps else float("nan")

            key = (round(ev_prog, 1), -(lap_mean if laps else 1e9))
            if best_key is None or key >= best_key:
                best_key = key
                agent.save(os.path.join(run_dir, "model_best.npz"))

            w_me.writerow([it, ep_no, total_steps, round(tr_rew, 3), round(tr_prog, 2), round(tr_lap, 1), round(ev_rew, 3),
                           round(ev_prog, 2), round(ev_lap, 1), "" if not laps else round(lap_mean, 3),
                           "" if not laps else round(lap_best, 3), round(st["entropy"], 4), round(st["kl"], 5),
                           round(st["policy_loss"], 5), round(st["value_loss"], 5), round(st["clip_fraction"], 4),
                           round(time.time() - t0, 1)])
            for f in (f_tr, f_ev, f_ep, f_me):
                f.flush()

            for k, v in zip(curves, (tr_rew, ev_rew, tr_prog, ev_prog, tr_lap, st["entropy"], st["kl"])):
                curves[k].append(v)
            if viewer:
                viewer.update_curves(curves)

            lap_txt = f"{lap_mean:5.2f}s" if laps else "  -  "
            print(f"iter {it:3d} | ep {ep_no:5d} | train rew {tr_rew:7.1f} prog {tr_prog:5.1f}% lap {tr_lap:4.0f}% | "
                  f"eval prog {ev_prog:5.1f}% lap {ev_lap:4.0f}% time {lap_txt} | H {st['entropy']:.2f} KL {st['kl']:.4f} | "
                  f"{(time.time() - t0) / 60:4.1f}min", flush=True)

            write_status(run_dir, state="running", iteration=it, max_iterations=max_iters, episodes=ep_no,
                         elapsed_s=round(time.time() - t0, 1))
            if os.path.exists(os.path.join(run_dir, "STOP")):      # 대시보드의 '학습 중지'
                stop_reason = "stopped_by_user"
                break
            if ep_no >= hp["term_cond_max_episodes"]:
                break
            if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= hp["term_cond_avg_score"]:
                stop_reason = "term_cond_avg_score"
                break
            if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
                stop_reason = "max_minutes"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\n중단됨. 지금까지의 결과를 저장합니다.")
    except RewardFunctionError as e:
        stop_reason = "reward_error"
        print(f"\n[보상함수 오류] {e}")
    finally:
        for f in (f_tr, f_ev, f_ep, f_me):
            f.close()

    agent.save(os.path.join(run_dir, "model_final.npz"))
    write_status(run_dir, state="finished", stop_reason=stop_reason, iteration=it, max_iterations=max_iters,
                 episodes=ep_no, elapsed_s=round(time.time() - t0, 1))
    print(f"\n종료 사유: {stop_reason} | {it} iterations, {ep_no} episodes, {total_steps} steps, {(time.time() - t0) / 60:.1f}분")
    try:
        from miniracer.analysis import make_report
        out = make_report(run_dir)
        print(f"분석 리포트: {out}")
    except Exception as e:     # 리포트 실패가 학습 결과 저장을 막지 않도록
        print(f"[알림] 리포트 생성 실패: {type(e).__name__}: {e}  (python analyze.py --run {os.path.basename(run_dir)} 로 다시 시도)")
    print(f"저장 위치: {run_dir}")
    if viewer is not None:
        try:
            viewer.save(os.path.join(run_dir, "live_view.png"))
        except Exception:
            pass
        import matplotlib.pyplot as plt
        print("창을 닫으면 종료됩니다.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
