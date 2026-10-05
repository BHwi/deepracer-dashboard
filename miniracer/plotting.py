"""그래픽 공통 모듈: 한글 글꼴 자동 설정, 트랙 그리기, CSV 읽기."""
import csv

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.patches import Polygon

# ----------------------------------------------------------------------------- 한글 글꼴 자동 설정
_KO_FONTS = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Nanum Gothic",
             "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR", "UnDotum", "Gulim"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_ko = next((f for f in _KO_FONTS if f in _available), None)
if _ko:
    rcParams["font.family"] = _ko
HAS_KO = _ko is not None
rcParams["axes.unicode_minus"] = False     # 한글 폰트에서 '-' 기호 깨짐 방지


def T(ko, en):
    """한글 글꼴이 있으면 한글, 없으면 영어 라벨."""
    return ko if HAS_KO else en


BG = "#00C389"        # AWS 트랙 이미지 색감
ROAD = "#232F3E"
EDGE = "#FFFFFF"
DASH = "#FF9900"


def draw_track(ax, track, title=None):
    ax.set_facecolor(BG)
    ax.add_patch(Polygon(track.outer, closed=True, fc=ROAD, ec=EDGE, lw=2, zorder=1))
    ax.add_patch(Polygon(track.inner, closed=True, fc=BG, ec=EDGE, lw=2, zorder=2))
    c = np.vstack([track.center, track.center[:1]])
    ax.plot(c[:, 0], c[:, 1], ls="--", color=DASH, lw=1.0, zorder=3)
    ax.plot(*track.center[0], marker="s", color="white", ms=6, zorder=3)        # 출발선
    ax.set_xlim(track.bounds[0], track.bounds[1])
    ax.set_ylim(track.bounds[2], track.bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def load_csv(path):
    """CSV 를 {열이름: 리스트} 로 읽는다 (숫자는 float 로 변환, 문자열은 그대로)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    if not rows:
        return out
    for k in rows[0].keys():
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col, dtype=object)
    return out
