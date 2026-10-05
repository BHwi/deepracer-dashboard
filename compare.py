"""여러 실험을 겹쳐서 비교한다 (변인 1개 대조 실험용).

  python compare.py baseline speed_x2 lr_high
"""
import argparse
import os

from miniracer.analysis import make_compare

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="runs/ 아래 실험 이름들")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dirs = [r if os.path.isdir(r) else os.path.join(HERE, "runs", r) for r in args.runs]
    out, table = make_compare(dirs, args.out)
    print(table)
    print("그림:", out)
