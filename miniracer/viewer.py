"""실시간 뷰어 (matplotlib).

왼쪽: 트랙 + 차량 + 주행 궤적(색 = 그 스텝의 보상) + 레이 센서(차가 '보는' 것)
오른쪽: iteration 별 보상 / 진행률·완주율 / 엔트로피 / KL
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from .plotting import T, draw_track

CAR_VIS_SCALE = 3.5    # 화면에서만 차를 키워서 그림 (물리 크기는 그대로)


class Viewer:
    def __init__(self, env, title="MiniRacer", show_rays=True, stride=2):
        self.env, self.show_rays, self.stride = env, show_rays, stride
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        gs = self.fig.add_gridspec(4, 2, width_ratios=[1.35, 1])
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.axes = [self.fig.add_subplot(gs[i, 1]) for i in range(4)]
        draw_track(self.ax_map, env.track, T("트랙: ", "track: ") + env.track.name + T("  (스페이스: 일시정지)", "  (space: pause)"))

        self.car_patch = Polygon(self._car_xy(), closed=True, fc="#FF4F00", ec="white", lw=1.5, zorder=6)
        self.ax_map.add_patch(self.car_patch)
        self.trail = LineCollection([], cmap="viridis", lw=2.5, zorder=4)
        self.ax_map.add_collection(self.trail)
        self.rays = LineCollection([], colors="#FFE066", lw=0.8, alpha=0.7, zorder=5)
        self.ax_map.add_collection(self.rays)
        self.info = self.ax_map.text(0.01, 0.99, "", transform=self.ax_map.transAxes, va="top", ha="left",
                                     fontsize=10, family="monospace", bbox=dict(fc="white", alpha=0.85, ec="none"))
        self._pts, self._rews, self._n = [], [], 0
        self._setup_curves()
        self.paused = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _car_xy(self):
        e = self.env
        p = np.array(e.car_polygon())
        c = np.array([e.x, e.y])
        return c + (p - c) * CAR_VIS_SCALE

    def _setup_curves(self):
        a0, a1, a2, a3 = self.axes
        a0.set_ylabel(T("평균 보상", "Mean reward"))
        a1.set_ylabel(T("진행률·완주율 %", "Progress / laps %"))
        a2.set_ylabel(T("엔트로피", "Entropy"))
        a3.set_ylabel("KL")
        a3.set_xlabel("iteration")
        self.l_tr_rew, = a0.plot([], [], color="#1f77b4", lw=1.8, label=T("학습", "train"))
        self.l_ev_rew, = a0.plot([], [], color="#ff7f0e", lw=1.8, label=T("평가", "eval"))
        self.l_tr_prog, = a1.plot([], [], color="#2ca02c", lw=1.8, label=T("진행률(학습)", "progress (train)"))
        self.l_ev_prog, = a1.plot([], [], color="#ff7f0e", lw=1.8, label=T("진행률(평가)", "progress (eval)"))
        self.l_tr_lap, = a1.plot([], [], color="#9467bd", lw=1.2, ls="--", label=T("완주율(학습)", "laps done (train)"))
        self.l_ent, = a2.plot([], [], color="#9467bd", lw=1.8)
        self.l_kl, = a3.plot([], [], color="#d62728", lw=1.8)
        a0.legend(loc="upper left", fontsize=8)
        a1.legend(loc="lower right", fontsize=7)
        a1.set_ylim(0, 105)
        for ax in self.axes:
            ax.grid(alpha=0.3)

    # ------------------------------------------------------------------ 갱신
    def begin_episode(self):
        self._pts, self._rews, self._n = [(self.env.x, self.env.y)], [], 0
        self.trail.set_segments([])

    def update_step(self, reward, label=""):
        env = self.env
        self._pts.append((env.x, env.y))
        self._rews.append(reward)
        self._n += 1
        if env.done or self._n % self.stride == 0:
            self._draw(label)

    def _draw(self, label):
        env = self.env
        self.car_patch.set_xy(self._car_xy())
        if len(self._pts) > 1:
            p = np.array(self._pts)
            self.trail.set_segments(np.stack([p[:-1], p[1:]], axis=1))
            self.trail.set_array(np.array(self._rews))
            self.trail.set_clim(0, max(1e-3, max(self._rews)))
        if self.show_rays:
            ex, ey = env.ray_ends
            self.rays.set_segments([[(env.x, env.y), (a, b)] for a, b in zip(ex, ey)])
        self.info.set_text(
            f"{label}\nstep {env.steps:4d}   speed {env.speed:4.2f} m/s   steer {env.steer:+5.1f}°\n"
            f"dist_center {env.params['distance_from_center']:.2f} m   progress {env.progress:5.1f}%\n"
            f"reward {env.last_reward:5.2f}   {env.status}")
        self._flush()

    def update_curves(self, m):
        it = np.arange(1, len(m["train_reward"]) + 1)
        self.l_tr_rew.set_data(it, m["train_reward"])
        self.l_ev_rew.set_data(it, m["eval_reward"])
        self.l_tr_prog.set_data(it, m["train_progress"])
        self.l_ev_prog.set_data(it, m["eval_progress"])
        self.l_tr_lap.set_data(it, m["train_lap_rate"])
        self.l_ent.set_data(it, m["entropy"])
        self.l_kl.set_data(it, m["kl"])
        for i, ax in enumerate(self.axes):
            ax.relim()
            ax.autoscale_view(scaley=(i != 1))
        self._flush()

    def _flush(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        while self.paused:
            plt.pause(0.1)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused

    def save(self, path):
        self.fig.savefig(path, dpi=110)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
