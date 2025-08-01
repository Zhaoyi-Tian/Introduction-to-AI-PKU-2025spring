import argparse
import subprocess
import numpy as np
import time

PYTHON_PATH = "D:/HuaweiMoveData/Users/50376/Desktop/Ai引论/2025-AI-intro-lab4-release-v1.6/.venv/Scripts/python.exe"

parser = argparse.ArgumentParser()
parser.add_argument("--q", choices=["q1", "q2", "q3", "all"], default="all")
args = parser.parse_args()


def programcall(cmd: str, timeout: float=600):
    ret = subprocess.check_output(cmd, shell=True, timeout=timeout)
    ret = str(ret, encoding="utf-8")
    return ret.splitlines()

def filterq1(outputlines):
    ret = list(filter(lambda line: "Error is" in line, outputlines))
    return float(ret[0].split()[-1])

def filterq2(outputlines):
    res = outputlines[0].split()
    return bool(res[0]), np.array([float(res[1]), float(res[2])])

def filterq3(outputlines):
    res = outputlines[0].split()
    if res[0] == "Failed!":
        return False, int(res[2])
    return True, int(res[2])

def scorer(acc, standardacc, zeroacc=0.1, fullacc=1):
    '''
    stardardacc代表90分时的成绩
    zeroacc是0分线
    fullacc是满分线
    '''
    if acc < standardacc:
        a = (90-0)/(standardacc-zeroacc)
        b = 0 - zeroacc * a
    else:
        a = (100-90)/(fullacc - standardacc)
        b = 100 - fullacc * a
    y = a*acc + b
    return max(min(y, 100), 0)

def scorer_q3(cnt, standardcnt, zerocnt, fullcnt):
    '''
    stardardcnt代表90分时的成绩
    zerocnt是0分线
    fullcnt是满分线
    '''
    if cnt > standardcnt:
        a = (90-0)/(standardcnt-zerocnt)
        b = 0 - zerocnt * a
    else:
        a = (100-90)/(fullcnt - standardcnt)
        b = 100 - fullcnt * a
    y = a*cnt + b
    return max(min(y, 100), 0)

def q1():
    print("q1 Localization")
    errors = []
    for i in range(15):
        output = programcall(f"{PYTHON_PATH} taskLocalization.py --no_render --test_idx {i}")
        print(output)
        errors.append(filterq1(output))
    return scorer(np.exp(-np.mean(errors)), 0.90, 0.0, 0.95)

def q2():
    print("q2 PDControl")
    answers = np.load('data_q2/checkpoints.npy')
    res = []
    for i in range(answers.shape[0]):
        output = programcall(f"{PYTHON_PATH} taskPDControl.py --no_render --test_idx {i}")
        complete, checkpoint = filterq2(output)
        if not complete:
            return 0
        res.append(checkpoint)
    err = np.linalg.norm(np.array(res) - answers)
    return scorer(np.exp(-err), 0.95, 0.9, 0.99)

def q3():
    print("q3 Planning")
    food_cnts = [  25,   25,   25,   25,   50,     50]
    weights   = [0.22, 0.22, 0.22, 0.22,  0.06,  0.06]
    stdcnt    = [1500, 3000, 1500, 3500, 18000, 18000]
    zerocnt   = [5000, 5000, 5000, 5000, 20000, 20000]
    fullcnt   = [1000, 2500, 1000, 3000, 10000, 10000]
    score = 0.
    # 规定步数内食物全部吃掉，拿0.90的分数，所用步数决定后0.10的分数
    for i in range(6):
        output = programcall(f"{PYTHON_PATH} taskPlanning.py --no_render --test_idx {6+i} --max_step {zerocnt[i]}")
        flag, cnt = filterq3(output)
        if flag is False:
            task_score =  (90 * np.sqrt(cnt / food_cnts[i]))
        else:
            task_score = (90 + scorer_q3(cnt, stdcnt[i], zerocnt[i], fullcnt[i]) * 0.1)
        print(f"Case {i}, score={task_score:.4f}")
        score += weights[i] * task_score
    return score

if __name__ == "__main__":
    if args.q == "all":
        for q in [q1, q2, q3]:
            print(f"score {q():.0f}")
    else:
        print(f"score {eval(args.q)():.0f}")