"""AWS DeepRacer 커뮤니티 저장소에서 트랙(.npy)을 내려받아 tracks/ 에 저장한다.

  python fetch_tracks.py                      # 추천 트랙 (짧고 폭이 다양한 것)
  python fetch_tracks.py Monaco Singapore     # 이름을 직접 지정
  python fetch_tracks.py --list               # 저장소의 트랙 목록(웹 페이지 파싱)

출처: https://github.com/aws-deepracer-community/deepracer-race-data  (raw_data/tracks/npy)
주의: 해당 저장소에서 라이선스 파일을 찾지 못했습니다. 다른 곳에 재배포하기 전에 확인하세요.
"""
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/aws-deepracer-community/deepracer-race-data/main/raw_data/tracks/npy/"
RECOMMENDED = ["Oval_track", "Vegas_track", "Canada_Training", "reinvent_base", "reInvent2019_wide"]


def main(argv):
    if "--list" in argv:
        url = "https://github.com/aws-deepracer-community/deepracer-race-data/tree/main/raw_data/tracks/npy"
        html = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
        names = sorted(set(re.findall(r"raw_data/tracks/npy/([A-Za-z0-9_.\-]+)\.npy", html)))
        print(f"{len(names)}개:", ", ".join(names))
        return
    names = [a for a in argv if not a.startswith("-")] or RECOMMENDED
    os.makedirs(os.path.join(HERE, "tracks"), exist_ok=True)
    for n in names:
        n = n[:-4] if n.endswith(".npy") else n
        dest = os.path.join(HERE, "tracks", n + ".npy")
        try:
            urllib.request.urlretrieve(BASE + n + ".npy", dest)
            print("받음:", n)
        except Exception as e:
            print(f"실패: {n} ({e})")
    print("\n주의: 장거리 트랙(약 50m 이상)은 학습 시간이 오래 걸리고, 직선 트랙(Straight_track)처럼 닫힌 루프가 아닌 트랙은 쓸 수 없습니다.")


if __name__ == "__main__":
    main(sys.argv[1:])
