import random, json
from env import TaskEnv

def train_q_learning(episodes=300):
    Q = {}
    alpha, gamma, eps = 0.1, 0.9, 0.2
    env = TaskEnv()

    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            st = tuple(s)
            if random.random() < eps:
                a = random.randint(0,3)
            else:
                a = max(Q.get(st,{i:0 for i in range(4)}), key=Q.get(st,{i:0 for i in range(4)}).get)

            ns,r,done = env.step(a)
            nt = tuple(ns)

            Q.setdefault(st,{i:0 for i in range(4)})
            Q.setdefault(nt,{i:0 for i in range(4)})

            Q[st][a] += alpha*(r + gamma*max(Q[nt].values()) - Q[st][a])
            s = ns
    return Q

def save_q(Q):
    json.dump({",".join(map(str,k)):v for k,v in Q.items()}, open("q.json","w"))

def load_q():
    try:
        data = json.load(open("q.json"))
        return {tuple(map(int,k.split(","))):v for k,v in data.items()}
    except:
        return {}