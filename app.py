from fastapi import FastAPI
from env import TaskEnv
from agent import train_q_learning, save_q, load_q
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt, io, base64

app = FastAPI()
env = TaskEnv()

Q = load_q()
dqn = DQNAgent(); dqn.load()

@app.get("/")
def home():
    return {"msg":"Scheduler API","endpoints":["/reset","/step?action=0","/train_q","/train_dqn","/evaluate","/curve"]}

@app.get("/reset")
def reset():
    return {"state":env.reset(),"reward":0,"done":False}

@app.get("/step")
def step(action:int):
    s,r,d=env.step(action)
    return {"state":s,"reward":r,"done":d}

@app.get("/train_q")
def train_q():
    global Q
    Q=train_q_learning(300)
    save_q(Q)
    return {"msg":"Q trained"}

@app.get("/train_dqn")
def train_dqn():
    global dqn
    rewards=[]
    for _ in range(200):
        s=env.reset(); done=False; total=0
        while not done:
            a=dqn.select_action(s)
            ns,r,done=env.step(a)
            dqn.train_step(s,a,r,ns,done)
            s=ns; total+=r
        rewards.append(total)
    dqn.save()
    return {"msg":"DQN trained","rewards":rewards}

def greedy(s):
    for i in range(4):
        if s[1+i*2]!=0: return i
    return 0

def q_policy(s):
    t=tuple(s)
    return max(Q[t],key=Q[t].get) if t in Q else greedy(s)

def dqn_policy(s):
    return dqn.select_action(s)

def run(p):
    s=env.reset(); total=0; done=False
    while not done:
        a=p(s)
        s,r,done=env.step(a)
        total+=r
    return total

@app.get("/evaluate")
def eval():
    g=sum(run(greedy) for _ in range(10))/10
    q=sum(run(q_policy) for _ in range(10))/10
    d=sum(run(dqn_policy) for _ in range(10))/10
    return {"greedy":g,"q":q,"dqn":d}

@app.get("/curve")
def curve():
    rewards=[run(dqn_policy) for _ in range(50)]
    plt.plot(rewards)
    buf=io.BytesIO(); plt.savefig(buf,format="png")
    return {"img":base64.b64encode(buf.getvalue()).decode()}
