"""가짜 학습기: TrainingMetrics.json 을 DRfC 와 같은 필드로 천천히 써서 학습이 진행되는 것처럼 보이게 한다."""
import json, math, os, random, signal, sys, time

mode = sys.argv[1]
D = os.environ["DR_DIR"]
prefix = os.environ["DR_LOCAL_S3_MODEL_PREFIX"]
bucket = os.path.join(D, "data", "minio", os.environ.get("DR_LOCAL_S3_BUCKET", "bucket"))
state = os.path.join(D, ".mock_state")
step = float(os.environ.get("MOCK_SPEED", "0.08"))
exp = os.environ.get("DR_EXPERIMENT_NAME")
hp = {}
if exp:
    try:
        hp = json.load(open(os.path.join(D, "experiments", exp, "custom_files", "hyperparameters.json")))
    except Exception:
        pass
max_eps = int(hp.get("term_cond_max_episodes", 1000)); per = int(hp.get("num_episodes_between_training", 20))
n_eval = int(os.environ.get("DR_TRAIN_MIN_EVAL_TRIALS", "5"))
img = f"{os.environ.get('DR_SIMAPP_SOURCE', 'x')}:{os.environ.get('DR_SIMAPP_VERSION', 'x')}"
rid = os.environ.get("DR_RUN_ID", "0")
random.seed(abs(hash(prefix)) % 1000)


def containers(on):
    p = os.path.join(state, "containers.txt")
    if not on:
        if os.path.exists(p): os.remove(p)
        return
    open(os.path.join(state, "prefix.txt"), "w").write(prefix)          # 'docker inspect' 가짜가 시뮬레이터의 MODEL_S3_PREFIX 로 돌려준다
    if mode == "eval":      # 실제 DRfC: 평가 스택은 deepracer-eval-<번호> 이고 시뮬레이터만 있다
        rows = [f"s3_minio.1.abc|minio/minio:latest|Up 3 hours", f"deepracer-eval-{rid}_robomaker.1.mock|{img}|Up 20 seconds"]
    else:
        rows = [f"s3_minio.1.abc|minio/minio:latest|Up 3 hours", f"deepracer-{rid}-algo-1-mock|{img}|Up 20 seconds",
                f"deepracer-{rid}_robomaker.1.mock|{img}|Up 20 seconds", f"deepracer-{rid}_rl_coach.1.mock|{img}|Up 20 seconds"]
    open(p, "w").write("\n".join(rows) + "\n")


def log(msg):
    open(os.path.join(state, "mock.log"), "a").write(msg + "\n")


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"metrics": rows, "version": "1.0"}, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def bye(*a):
    containers(False); sys.exit(0)


signal.signal(signal.SIGTERM, bye)
containers(True)
open(os.path.join(state, "mock.log"), "w").write(f"[mock] {mode} {prefix} (이 로그는 가짜입니다)\n")

if mode == "train":
    mpath = os.path.join(bucket, prefix, "metrics", "TrainingMetrics.json")
    rows, t0 = [], time.time()
    meta = os.path.join(bucket, "custom_files", "model_metadata.json")
    for ep in range(1, max_eps + 1):
        skill = 1 - math.exp(-ep / (max_eps * 0.3))
        prog = max(2, min(100, 100 * skill * random.uniform(0.75, 1.1)))
        lap = prog >= 99.5 or random.random() < skill * 0.7
        if lap: prog = 100.0
        rows.append({"reward_score": round(prog * random.uniform(2.5, 3.5), 2), "metric_time": int(time.time() * 1000), "start_time": int(t0 * 1000),
                     "elapsed_time_in_milliseconds": int((12 + 25 * (1 - skill)) * 1000 * random.uniform(0.9, 1.1)) if lap else int(random.uniform(3000, 15000)),
                     "episode": ep, "trial": (ep - 1) % per + 1, "phase": "training", "completion_percentage": round(prog, 2),
                     "episode_status": "Lap complete" if lap else random.choice(["Off track", "Off track", "Off track", "Crashed"])})
        log(f"Training> Name=main_level/agent, Worker=0, Episode={ep}, Total reward={rows[-1]['reward_score']}, Steps={int(prog * 3)}, Training iteration={(ep - 1) // per}")
        if ep % per == 0:
            for k in range(n_eval):
                ep_lap = random.random() < min(0.97, skill * 1.05)
                rows.append({"reward_score": round(random.uniform(150, 350) * skill, 2), "metric_time": int(time.time() * 1000), "start_time": int(t0 * 1000),
                             "elapsed_time_in_milliseconds": int((11 + 20 * (1 - skill)) * 1000 * random.uniform(0.95, 1.05)) if ep_lap else int(random.uniform(4000, 14000)),
                             "episode": ep, "trial": k + 1, "phase": "evaluation", "completion_percentage": 100.0 if ep_lap else round(random.uniform(30, 90), 2),
                             "episode_status": "Lap complete" if ep_lap else "Off track"})
            write(mpath, rows)
            if os.path.isdir(os.path.join(bucket, prefix, "model")) and os.path.exists(meta):
                open(os.path.join(bucket, prefix, "model", "model_metadata.json"), "w").write(open(meta).read())
                open(os.path.join(bucket, prefix, "model", f"model_{ep // per}.pb"), "w").write("mock")
        time.sleep(step)
    write(mpath, rows)
    log("[mock] 학습 완료")
    containers(False)
else:
    epath = os.path.join(bucket, prefix, "metrics", "evaluation", f"evaluation-{time.strftime('%Y%m%d%H%M%S')}.json")
    trials = int(os.environ.get("DR_EVAL_NUMBER_OF_TRIALS", "3"))
    rows = []
    for k in range(trials):
        time.sleep(step * 15)
        ok = random.random() < 0.8
        rows.append({"reward_score": round(random.uniform(200, 340), 2), "elapsed_time_in_milliseconds": int(random.uniform(11000, 16000)) if ok else int(random.uniform(4000, 12000)),
                     "episode": k + 1, "trial": k + 1, "phase": "evaluation", "completion_percentage": 100.0 if ok else round(random.uniform(40, 95), 2),
                     "episode_status": "Lap complete" if ok else "Off track"})
        write(epath, rows)
    containers(False)
