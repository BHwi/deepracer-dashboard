"""학습 결과 분석: 실험 1개 리포트(make_report), 여러 실험 비교(make_compare).

실제 DeepRacer 로그 분석과 같은 관점으로 본다.
  - 학습 곡선(보상·진행률·완주율·랩타임) / 알고리즘 내부 지표(엔트로피·KL·손실)
  - 스텝별 trace 로 본 주행: 보상 히트맵, 구간별 속도·조향, 행동 선택 분포, 에피소드 종료 사유
"""
import json
import os

import numpy as np
from matplotlib.figure import Figure

from .plotting import T, draw_track, load_csv
from .track import Track, resolve_track

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(HERE, "tracks")


def _load_run(run_dir):
    info = json.load(open(os.path.join(run_dir, "run_info.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(run_dir, "model_metadata.json"), encoding="utf-8"))
    acts = sorted(meta["action_space"], key=lambda a: a.get("index", 0))
    return {
        "info": info,
        "actions": [(float(a["steering_angle"]), float(a["speed"])) for a in acts],
        "metrics": load_csv(os.path.join(run_dir, "metrics.csv")),
        "episodes": load_csv(os.path.join(run_dir, "episodes.csv")),
        "trace_train": load_csv(os.path.join(run_dir, "trace_train.csv")),
        "trace_eval": load_csv(os.path.join(run_dir, "trace_eval.csv")),
    }


def _load_track(info):
    for cand in (info.get("track_file"), info.get("track")):
        if not cand:
            continue
        try:
            return Track(resolve_track(cand, TRACKS_DIR))
        except FileNotFoundError:
            continue
    return None


def _best_iteration(m):
    """train.py 의 best 모델 선택 기준과 같은 기준으로 best iteration 을 찾는다."""
    best, best_key = None, None
    for i in range(len(m["iteration"])):
        lap = m["eval_lap_time_mean"][i]
        key = (round(min(100.0, float(m["eval_progress"][i])), 1), -(1e9 if np.isnan(lap) else float(lap)))
        if best_key is None or key >= best_key:
            best, best_key = int(m["iteration"][i]), key
    return best


def _best_eval_episode(trace_eval, iteration):
    """해당 iteration 의 평가 에피소드 중 가장 잘 달린 것의 trace (진행률 최대, 같으면 스텝 적은 것)."""
    mask = trace_eval["iteration"] == iteration
    if not mask.any():
        return None
    eps = np.unique(trace_eval["episode"][mask])
    best, best_key = None, None
    for e in eps:
        sel = mask & (trace_eval["episode"] == e)
        key = (round(float(trace_eval["progress"][sel].max()), 1), -int(sel.sum()))
        if best_key is None or key > best_key:
            best, best_key = sel, key
    return best


def _line(ax, x, y, **kw):
    ax.plot(x, y, **kw)


def make_report(run_dir, out_path=None):
    r = _load_run(run_dir)
    m, ep, tt, te, acts = r["metrics"], r["episodes"], r["trace_train"], r["trace_eval"], r["actions"]
    track = _load_track(r["info"])
    it = m["iteration"]
    name = os.path.basename(os.path.normpath(run_dir))

    fig = Figure(figsize=(18, 15))
    ax = fig.subplots(4, 3)
    fig.suptitle(f"MiniRacer report - {name}   (track: {r['info']['track']})", fontsize=14)

    # ---------------- 1행: 학습 곡선
    a = ax[0, 0]
    _line(a, it, m["train_reward"], color="#1f77b4", label=T("학습", "train"))
    _line(a, it, m["eval_reward"], color="#ff7f0e", label=T("평가", "eval"))
    a.set_title(T("에피소드 평균 보상 (보상함수가 다르면 비교 불가)", "Mean episode reward (not comparable across reward functions)"), fontsize=10)
    a.legend()

    a = ax[0, 1]
    _line(a, it, m["train_progress"], color="#2ca02c", label=T("진행률 (학습)", "progress (train)"))
    _line(a, it, m["eval_progress"], color="#ff7f0e", label=T("진행률 (평가)", "progress (eval)"))
    _line(a, it, m["train_lap_rate"], color="#9467bd", ls="--", label=T("완주율 (학습)", "laps done (train)"))
    a.set_ylim(0, 105)
    a.set_title(T("진행률 · 완주율 (%)", "Progress / lap completion (%)"), fontsize=10)
    a.legend(fontsize=8, loc="lower right")

    a = ax[0, 2]
    lap_mean, lap_best = m["eval_lap_time_mean"], m["eval_lap_time_best"]
    if np.isfinite(lap_mean).any():
        _line(a, it, lap_mean, color="#ff7f0e", marker="o", ms=3, label=T("평균", "mean"))
        _line(a, it, lap_best, color="#d62728", marker="o", ms=3, ls=":", label=T("최고", "best"))
        a.legend()
    else:
        a.text(0.5, 0.5, T("평가 주행에서 완주한 적이 없습니다", "no completed lap in evaluation"), ha="center", va="center", transform=a.transAxes)
    a.set_title(T("평가 랩타임 (초, 완주한 경우만)", "Eval lap time (s, completed laps only)"), fontsize=10)

    # ---------------- 2행: 알고리즘 내부 지표
    a = ax[1, 0]
    _line(a, it, m["entropy"], color="#9467bd")
    a.set_title(T("엔트로피 (탐험 정도: 높음=이것저것 시도, 낮아짐=한 가지로 굳어짐)", "Entropy (high = exploring, falling = settling)"), fontsize=9)
    a = ax[1, 1]
    _line(a, it, m["kl"], color="#d62728")
    a.set_title(T("KL 다이버전스 (한 번 갱신에 정책이 바뀐 폭, 급등하면 위험)", "KL divergence (policy change per update; spikes = unstable)"), fontsize=9)
    a = ax[1, 2]
    _line(a, it, m["policy_loss"], color="#1f77b4", label=T("정책 손실", "policy loss"))
    a.set_ylabel(T("정책 손실", "policy loss"), color="#1f77b4")
    a2 = a.twinx()
    _line(a2, it, m["value_loss"], color="#2ca02c", label=T("가치 손실", "value loss"))
    a2.set_ylabel(T("가치 손실", "value loss"), color="#2ca02c")
    a.set_title(T("손실 (loss)", "Losses"), fontsize=10)

    # ---------------- 3행: 트랙 위에서 본 주행
    a = ax[2, 0]
    if track is not None and tt and len(tt.get("x", [])):
        draw_track(a, track, T("보상 히트맵 (학습 후반 주행 기록)", "Reward heatmap (late training trace)"))
        its = np.unique(tt["iteration"])
        late = tt["iteration"] >= its[int(len(its) * 0.6)] if len(its) > 2 else np.ones(len(tt["x"]), bool)
        idx = np.flatnonzero(late)
        if len(idx) > 60000:
            idx = np.random.default_rng(0).choice(idx, 60000, replace=False)
        sc = a.scatter(tt["x"][idx], tt["y"][idx], c=tt["reward"][idx], s=3, cmap="plasma", zorder=5)
        fig.colorbar(sc, ax=a, fraction=0.04)
    else:
        a.text(0.5, 0.5, "no trace / track", ha="center", va="center", transform=a.transAxes)

    best_it = _best_iteration(m)
    sel = _best_eval_episode(te, best_it) if te else None
    a = ax[2, 1]
    if track is not None and sel is not None:
        draw_track(a, track, T(f"best 모델의 평가 주행 (iteration {best_it}) - 색: 속도", f"best model eval run (iter {best_it}) - color: speed"))
        sc = a.scatter(te["x"][sel], te["y"][sel], c=te["speed"][sel], s=8, cmap="viridis", zorder=5)
        fig.colorbar(sc, ax=a, fraction=0.04)
    else:
        a.text(0.5, 0.5, "no trace / track", ha="center", va="center", transform=a.transAxes)

    def usage(axis, values, title):
        if not tt or not len(tt.get("iteration", [])):
            axis.text(0.5, 0.5, "no trace", ha="center", va="center", transform=axis.transAxes)
            return
        its_u = np.unique(tt["iteration"])
        act = tt["action"].astype(int)
        cmd = np.array([values(acts[i]) for i in act])
        for v in np.unique([values(x) for x in acts]):
            frac = [100.0 * np.mean(cmd[tt["iteration"] == i] == v) for i in its_u]
            axis.plot(its_u, frac, marker="o", ms=3, label=f"{v:g}")
        axis.set_ylim(0, 100)
        axis.legend(fontsize=8, ncol=3)
        axis.set_title(title, fontsize=10)

    usage(ax[2, 2], lambda x: x[1], T("선택한 속도 비율 (%, 학습 중 기록된 iteration)", "Share of chosen speeds (%)"))

    # ---------------- 4행: 구간별 주행, 종료 사유, 조향 선택
    a = ax[3, 0]
    if sel is not None:
        a.plot(te["progress"][sel], te["speed"][sel], color="#1f77b4", label=T("속도 (m/s)", "speed (m/s)"))
        a.set_ylabel(T("속도 (m/s)", "speed (m/s)"), color="#1f77b4")
        a2 = a.twinx()
        a2.plot(te["progress"][sel], te["steering_angle"][sel], color="#d62728", alpha=0.7, label=T("조향 (도)", "steer (deg)"))
        a2.set_ylabel(T("조향 (도)", "steer (deg)"), color="#d62728")
        a.set_xlabel(T("진행률 (%)", "progress (%)"))
    a.set_title(T("best 모델 평가 주행: 구간별 속도·조향", "Best eval run: speed & steering along the track"), fontsize=10)

    a = ax[3, 1]
    mask = ep["phase"] == "train"
    its_e = np.unique(ep["iteration"][mask])
    bottom = np.zeros(len(its_e))
    for status, color in (("lap_complete", "#2ca02c"), ("timeout", "#ff7f0e"), ("offtrack", "#d62728")):
        frac = np.array([100.0 * np.mean(ep["status"][mask & (ep["iteration"] == i)] == status) for i in its_e])
        a.bar(its_e, frac, bottom=bottom, color=color, label=status, width=1.0)
        bottom += frac
    a.set_ylim(0, 100)
    a.legend(fontsize=8, loc="lower right")
    a.set_title(T("에피소드 종료 사유 (%, 학습)", "Episode end reasons (%, train)"), fontsize=10)

    usage(ax[3, 2], lambda x: x[0], T("선택한 조향각 비율 (%)", "Share of chosen steering angles (%)"))

    for row in ax:
        for a in row:
            a.grid(alpha=0.3)
            if a.get_xlabel() == "" and a not in (ax[2, 0], ax[2, 1]):
                a.set_xlabel("iteration")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_path or os.path.join(run_dir, "report.png")
    fig.savefig(out_path, dpi=90)
    return out_path


def make_compare(run_dirs, out_path=None):
    """여러 실험을 겹쳐 그리고 요약표 문자열을 돌려준다. (보상 값은 보상함수가 다르면 비교할 수 없으므로 진행률·랩타임으로 비교)"""
    runs = [(os.path.basename(os.path.normpath(d)), _load_run(d)) for d in run_dirs]
    fig = Figure(figsize=(17, 5))
    ax = fig.subplots(1, 3)
    for name, r in runs:
        m = r["metrics"]
        ax[0].plot(m["iteration"], m["eval_progress"], label=name)
        ax[1].plot(m["iteration"], m["eval_lap_time_mean"], marker="o", ms=3, label=name)
        ax[2].plot(m["iteration"], m["train_lap_rate"], label=name)
    ax[0].set_title(T("평가 진행률 (%)", "Eval progress (%)"))
    ax[1].set_title(T("평가 평균 랩타임 (초)", "Eval mean lap time (s)"))
    ax[2].set_title(T("학습 완주율 (%)", "Train lap completion (%)"))
    ax[0].set_ylim(0, 105)
    ax[2].set_ylim(0, 105)
    for a in ax:
        a.set_xlabel("iteration")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_path or os.path.join(HERE, "runs", "compare_" + "_vs_".join(n for n, _ in runs)[:80] + ".png")
    fig.savefig(out_path, dpi=100)

    lines = [f"{'실험':24s} {'트랙':20s} {'iter':>5s} {'최종 평가진행률':>14s} {'최고 랩타임':>10s} {'첫 완주 iter':>12s} {'최종 엔트로피':>12s}"]
    for name, r in runs:
        m = r["metrics"]
        laps = m["eval_lap_time_best"]
        first = np.flatnonzero(m["eval_lap_rate"] > 0)
        best_lap = f"{np.nanmin(laps):.2f}s" if np.isfinite(laps).any() else "-"
        lines.append(f"{name:24s} {r['info']['track']:20s} {int(m['iteration'][-1]):5d} {m['eval_progress'][-1]:13.1f}% "
                     f"{best_lap:>10s} {(int(m['iteration'][first[0]]) if len(first) else '-'):>12} {m['entropy'][-1]:12.2f}")
    lines.append("")
    lines.append(T("※ 시드 1개의 결과입니다. 차이가 작으면 --seed 를 바꿔 반복해서 평균을 내세요.",
                   "* Single-seed results. If differences are small, repeat with other --seed values."))
    return out_path, "\n".join(lines)
