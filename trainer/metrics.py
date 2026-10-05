"""DRfC 가 만드는 TrainingMetrics.json / evaluation-*.json 읽기.

필드 이름은 커뮤니티 도구 deepracer-utils(deepracer/logs/metrics.py) 가 쓰는 것과 같다:
  reward_score, completion_percentage, episode_status, elapsed_time_in_milliseconds, episode, trial, phase('training'|'evaluation')
주의: 'iteration' 은 파일에 없어서, 학습 에피소드를 num_episodes_between_training 개씩 묶어 환산한다(추정).
"""
import json
import math
import os

import numpy as np

LAP = "Lap complete"
IN_PROGRESS = "In progress"


def outcome(r):
    """한 행의 결과 -> (완주 여부, 최종 상태가 기록되지 않아 '진행률 100%' 로 완주를 추정했는지).

    시뮬레이터가 쓰는 지표 파일에는 마지막 시도처럼 최종 상태가 기록되지 않은 행이 있다: 진행률은 100% 인데 episode_status 가 계속 'In progress' 로 남는다
    (시간이 지나도 갱신되지 않는다). 커뮤니티 도구 deepracer-utils 는 'Lap complete' 인 행만 완주로 세므로 그런 행은 완주로 세지 못한다.
    진행률 100% 는 한 바퀴를 다 돌았다는 뜻이므로 완주로 보되, 추정이라는 표시를 같이 돌려준다."""
    status = r.get("episode_status")
    if status == LAP:
        return True, False
    comp = r.get("completion_percentage")
    if status == IN_PROGRESS and comp is not None and comp >= 99.99:
        return True, True
    return False, False


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("metrics", [])
    except (OSError, ValueError):
        return None


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def parse_rows(data):
    """TrainingMetrics.json / evaluation-*.json 내용(bytes 또는 str)에서 metrics 목록을 꺼낸다. 형식이 맞지 않으면 None."""
    try:
        return json.loads(data).get("metrics", [])
    except (ValueError, AttributeError, TypeError):
        return None


def summarize_training(path, per_iter=20):
    rows = _read(path)
    if rows is None:
        return None
    s = summarize_training_rows(rows, per_iter)
    s["mtime"] = os.path.getmtime(path)
    return s


def summarize_training_rows(rows, per_iter=20):
    per_iter = max(1, int(per_iter))
    its = {}
    n_train = 0
    cur = 1
    for r in rows:
        phase = r.get("phase")
        if phase == "training":
            cur = n_train // per_iter + 1
            n_train += 1
        elif phase != "evaluation":
            continue
        d = its.setdefault(cur, {"tr_r": [], "tr_p": [], "tr_l": [], "ev_r": [], "ev_p": [], "ev_l": [], "ev_t": []})
        rew, comp = r.get("reward_score"), r.get("completion_percentage")
        lap = outcome(r)[0]
        if phase == "training":
            d["tr_r"].append(rew); d["tr_p"].append(comp); d["tr_l"].append(lap)
        else:
            d["ev_r"].append(rew); d["ev_p"].append(comp); d["ev_l"].append(lap)
            if lap and r.get("elapsed_time_in_milliseconds") is not None:
                d["ev_t"].append(r["elapsed_time_in_milliseconds"] / 1000.0)
    keys = sorted(its)
    out = {k: [] for k in ("iteration", "train_reward", "train_progress", "train_lap_rate", "eval_reward", "eval_progress",
                           "eval_lap_rate", "eval_lap_time_mean", "eval_lap_time_best", "n_train", "n_eval")}
    for k in keys:
        d = its[k]
        out["iteration"].append(k)
        out["train_reward"].append(_mean(d["tr_r"]))
        out["train_progress"].append(_mean(d["tr_p"]))
        out["train_lap_rate"].append(100.0 * _mean(d["tr_l"]) if d["tr_l"] else None)
        out["eval_reward"].append(_mean(d["ev_r"]))
        out["eval_progress"].append(_mean(d["ev_p"]))
        out["eval_lap_rate"].append(100.0 * _mean(d["ev_l"]) if d["ev_l"] else None)
        out["eval_lap_time_mean"].append(_mean(d["ev_t"]))
        out["eval_lap_time_best"].append(float(min(d["ev_t"])) if d["ev_t"] else None)
        out["n_train"].append(len(d["tr_r"]))
        out["n_eval"].append(len(d["ev_r"]))
    best = None
    best_key = None
    for i, k in enumerate(keys):
        p = out["eval_progress"][i]
        if p is None:
            continue
        lap = out["eval_lap_time_mean"][i]
        key = (round(min(100.0, p), 1), -(lap if lap is not None else 1e9))
        if best_key is None or key >= best_key:
            best, best_key = k, key
    return {"metrics": out, "episodes": n_train, "per_iter": per_iter, "best_iteration": best}


def _last_row_per_trial(rows):
    """같은 trial 번호의 행이 여러 개면 마지막 것만 남긴다 (같은 시도의 갱신 기록). trial 이 없는 행은 그대로 둔다."""
    out, pos = [], {}
    for r in rows:
        t = r.get("trial")
        if t is None:
            out.append(r)
        elif t in pos:
            out[pos[t]] = r
        else:
            pos[t] = len(out)
            out.append(r)
    return out


def summarize_evaluation_runs(files, running=False):
    """files: [(파일이름, rows)] -> 평가 실행별 요약 (최근 것이 먼저). running=True 면 가장 최근 파일의 'In progress' 는 진짜 진행 중으로 본다.
    평가가 끝난 뒤에도 진행률이 100% 미만인 'In progress' 는 '미완료'(Incomplete) 로 표시한다."""
    runs = []
    for k, (fn, rows) in enumerate(sorted(files, key=lambda x: x[0], reverse=True)):
        live = running and k == 0
        trials = []
        for r in _last_row_per_trial([x for x in (rows or []) if x.get("phase") in (None, "evaluation")]):
            ok, inferred = outcome(r)
            status = r.get("episode_status")
            if ok:
                status = LAP
            elif status == IN_PROGRESS and not live:
                status = "Incomplete"
            ms = r.get("elapsed_time_in_milliseconds")
            trials.append({"trial": r.get("trial"), "status": status, "progress": r.get("completion_percentage"), "inferred": inferred,
                           "time": (ms / 1000.0) if (ms is not None and ok) else None, "reward": r.get("reward_score")})
        laps = [t["time"] for t in trials if t["time"] is not None]
        runs.append({"file": fn, "trials": trials, "laps": sum(1 for t in trials if t["status"] == LAP), "n": len(trials), "inferred": sum(1 for t in trials if t["inferred"]),
                     "best": min(laps) if laps else None, "mean": float(np.mean(laps)) if laps else None,
                     "progress": _mean([t["progress"] for t in trials if t["progress"] is not None])})
    return runs


def summarize_evaluations(folder, running=False):
    """folder/evaluation-*.json 마다 한 번의 평가 실행으로 보고, 시도(trial)별 결과를 돌려준다."""
    files = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder), reverse=True):
            if fn.startswith("evaluation-") and fn.endswith(".json"):
                files.append((fn, _read(os.path.join(folder, fn)) or []))
    return summarize_evaluation_runs(files, running)


def match_videos(evals, videos, window=600):
    """저장된 영상(MP4)을 평가에 짝짓는다. 영상 파일 이름 규칙은 시뮬레이터가 정하고 알려져 있지 않으므로 '시각'으로 짝짓는다:
    영상은 '영상이 만들어진 시각과 가장 가까운 시각에 끝난 평가'의 것이다. 영상이 지표 파일보다 먼저 써지는지 나중에 써지는지 모르기 때문에 앞뒤 방향은 보지 않는다.
    (평가는 시작에만 1분 가까이 걸려서 연달아 돌려도 평가 끝 시각 사이가 영상-지표 시각 차이보다 훨씬 크다.)
      evals : [{file, end}]   end = 평가 지표 파일의 마지막 수정 시각 (epoch 초). 없으면 짝지을 수 없다.
      videos: [{key, modified, ...}]   modified = 영상 파일의 수정 시각 (epoch 초)
      window: 가장 가까운 평가와도 이 시간(초)보다 멀면 짝짓지 않는다.
    반환: ({평가 파일: [영상...]}, 어느 평가에도 짝지어지지 않은 영상들)"""
    by = {e["file"]: [] for e in evals}
    left = []
    ordered = sorted([e for e in evals if e.get("end")], key=lambda e: e["end"])
    for v in sorted(videos, key=lambda v: v.get("modified") or 0):
        m = v.get("modified")
        best = min(ordered, key=lambda e: abs(m - e["end"])) if (m is not None and ordered) else None
        if best is not None and abs(m - best["end"]) <= window:
            by[best["file"]].append(v)
        else:
            left.append(v)
    return by, left
