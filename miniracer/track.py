"""트랙 기하.

DeepRacer 커뮤니티 npy 포맷(N x 6: center_x, center_y, inner_x, inner_y, outer_x, outer_y)을 읽는다.

- 보상함수 params 에 필요한 값(distance_from_center, closest_waypoints, progress 등)을 계산
- 트랙 위/밖 판정을 빠르게 하기 위해 2.5cm 격자(occupancy grid)를 미리 만들어 둔다
- 차량의 "눈" 역할을 하는 레이(ray) 센서 거리 계산 (격자를 따라 걸어가며 첫 이탈 지점을 찾음)
"""
import os

import numpy as np
from matplotlib.path import Path

GRID_RES = 0.025   # 격자 해상도 (m)


class Track:
    def __init__(self, npy_path):
        self.name = os.path.splitext(os.path.basename(npy_path))[0]
        data = np.load(npy_path)
        center = data[:, 0:2]
        a, b = data[:, 2:4], data[:, 4:6]
        inner, outer = (a, b) if _poly_area(a) < _poly_area(b) else (b, a)

        # 닫힌 트랙이면 마지막 점(=첫 점) 제거
        if np.allclose(center[0], center[-1]):
            center, inner, outer = center[:-1], inner[:-1], outer[:-1]
        # 연속 중복점(길이 0 구간) 제거
        keep = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1) > 1e-9
        self.center, self.inner, self.outer = center[keep], inner[keep], outer[keep]
        self.n = len(self.center)

        seg = np.roll(self.center, -1, axis=0) - self.center
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum_len[-1])
        self.width = float(np.mean(np.linalg.norm(self.inner - self.outer, axis=1)))
        self.half_width = self.width / 2

        lo = self.outer.min(axis=0) - 0.5
        hi = self.outer.max(axis=0) + 0.5
        self.bounds = (lo[0], hi[0], lo[1], hi[1])
        self._build_grid()
        self._waypoints = [(float(x), float(y)) for x, y in np.vstack([self.center, self.center[:1]])]

    # ------------------------------------------------------------------ 격자
    def _build_grid(self):
        lo = np.array([self.bounds[0], self.bounds[2]])
        hi = np.array([self.bounds[1], self.bounds[3]])
        nx = int(np.ceil((hi[0] - lo[0]) / GRID_RES))
        ny = int(np.ceil((hi[1] - lo[1]) / GRID_RES))
        xs = lo[0] + (np.arange(nx) + 0.5) * GRID_RES
        ys = lo[1] + (np.arange(ny) + 0.5) * GRID_RES
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        in_outer = Path(self.outer).contains_points(pts)
        in_inner = Path(self.inner).contains_points(pts)
        self._grid = (in_outer & ~in_inner).reshape(ny, nx)
        self._lo, self._nx, self._ny = lo, nx, ny

    def on_track(self, x, y):
        """x, y: 스칼라 또는 배열. 트랙 위(경계선 안쪽)이면 True."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ix = np.floor((x - self._lo[0]) / GRID_RES).astype(int)
        iy = np.floor((y - self._lo[1]) / GRID_RES).astype(int)
        ok = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        out = np.zeros(ix.shape, dtype=bool)
        out[ok] = self._grid[iy[ok], ix[ok]]
        return out

    # ------------------------------------------------------------------ 중앙선 기준 위치
    def locate(self, x, y):
        """차량 위치를 중앙선에 투영해 보상함수 params 에 필요한 값을 만든다."""
        p = np.array([x, y])
        rel = p - self.center
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg_dir), 0.0, self.seg_len)
        foot = self.center + t[:, None] * self.seg_dir
        d = np.linalg.norm(foot - p, axis=1)
        i = int(np.argmin(d))
        cross = self.seg_dir[i, 0] * rel[i, 1] - self.seg_dir[i, 1] * rel[i, 0]
        return {
            "closest_waypoints": [i, (i + 1) % self.n],
            "distance_from_center": float(d[i]),
            "s": float(self.cum_len[i] + t[i]),
            "is_left_of_center": bool(cross > 0),
        }

    def heading_at(self, idx):
        idx %= self.n
        return float(np.degrees(np.arctan2(self.seg_dir[idx, 1], self.seg_dir[idx, 0])))

    def waypoints_list(self):
        return self._waypoints   # DeepRacer 규약: 첫 점 = 마지막 점

    # ------------------------------------------------------------------ 레이 센서
    def cast_rays(self, x, y, heading_deg, rel_angles_deg, max_range, step):
        """차량 중심에서 여러 방향으로 레이를 쏴서 트랙 경계까지의 거리를 돌려준다.

        returns (dist[R], end_x[R], end_y[R])
        """
        ang = np.radians(heading_deg + np.asarray(rel_angles_deg))
        d = np.arange(1, int(max_range / step) + 1) * step                  # (S,)
        px = x + np.cos(ang)[:, None] * d[None, :]                           # (R,S)
        py = y + np.sin(ang)[:, None] * d[None, :]
        off = ~self.on_track(px, py)
        first = np.argmax(off, axis=1)
        hit = off.any(axis=1)
        dist = np.where(hit, d[first], max_range)
        return dist, x + np.cos(ang) * dist, y + np.sin(ang) * dist


def _poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def list_tracks(tracks_dir):
    if not os.path.isdir(tracks_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(tracks_dir) if f.endswith(".npy"))


def resolve_track(name_or_path, tracks_dir):
    """트랙 이름(예: reInvent2019_track) 또는 npy 경로를 받아 파일 경로를 돌려준다."""
    if os.path.isfile(name_or_path):
        return name_or_path
    cand = os.path.join(tracks_dir, name_or_path if name_or_path.endswith(".npy") else name_or_path + ".npy")
    if os.path.isfile(cand):
        return cand
    have = ", ".join(list_tracks(tracks_dir)) or "(없음)"
    raise FileNotFoundError(
        f"트랙 '{name_or_path}' 을(를) 찾을 수 없습니다. 현재 tracks/ 에 있는 트랙: {have}\n"
        f"다른 트랙은 `python fetch_tracks.py` 로 내려받을 수 있습니다.")
