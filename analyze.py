"""실험 1개의 분석 리포트(report.png)를 (다시) 만든다.

  python analyze.py --run my_first
"""
import argparse
import os

from miniracer.analysis import make_report

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="runs/ 아래 실험 이름 (또는 폴더 경로)")
    args = p.parse_args()
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(HERE, "runs", args.run)
    print("저장:", make_report(run_dir))
