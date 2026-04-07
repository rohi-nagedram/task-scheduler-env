import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from env import TaskEnv
from agent import train_q_learning, save_q, load_q
from dqn_agent import DQNAgent

# ----------------------
# GLOBAL STATE
# ----------------------

env = TaskEnv()
dqn = DQNAgent()
Q = load_q()

leaderboard = []

# ----------------------
# UTIL
# ----------------------

def moving_avg(data, k=10):
    avg = []
    for i in range(len(data)):
        avg.append(sum(data[max(0, i-k):i+1]) / (i - max(0, i-k) + 1))
    return avg

# ----------------------
# ENV CONTROL
# ----------------------

def set_difficulty(level):
    global env
    env = TaskEnv(difficulty=level)
    return {"msg": f"Difficulty set to {level}"}, None

def reset():
    return {"state": env.reset(), "reward": 0, "done": False}, None

def step(action):
    s, r, d = env.step(int(action))
    return {"state": s, "reward": r, "done": d}, None

# ----------------------
# TRAIN Q
# ----------------------

def train_q():
    global Q
    Q = train_q_learning(200)
    save_q(Q)
    return {"msg": "Q-learning trained"}, None

# ----------------------
# REAL-TIME DQN TRAIN
# ----------------------

def train_dqn_realtime():
    rewards = []

    for ep in range(100):
        s = env.reset()
        done = False
        total = 0

        while not done:
            a = dqn.select_action(s)
            ns, r, done = env.step(a)

            dqn.train_step(s, a, r, ns, done)

            s = ns
            total += r

        rewards.append(total)

        # update graph every few steps
        if ep % 5 == 0:
            smoothed = moving_avg(rewards)

            fig = plt.figure()
            plt.plot(rewards, alpha=0.3)
            plt.plot(smoothed, linewidth=2)
            plt.title(f"Training (Episode {ep})")
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            yield {
                "msg": f"Training... episode {ep}",
                "latest_reward": total
            }, fig

    dqn.save()

    yield {
        "msg": "DQN training complete",
        "final_avg": sum(rewards)/len(rewards)
    }, fig

# ----------------------
# POLICIES
# ----------------------

def greedy_policy(state):
    for i in range(4):
        if state[1 + i*2] != 0:
            return i
    return 0

def dqn_policy(state):
    return dqn.select_action(state)

def run_episode(policy):
    s = env.reset()
    total = 0
    done = False

    while not done:
        a = policy(s)
        s, r, done = env.step(a)
        total += r

    return total

# ----------------------
# EVALUATE + LEADERBOARD
# ----------------------

def evaluate():
    global leaderboard

    greedy_scores = []
    dqn_scores = []

    for _ in range(20):
        greedy_scores.append(run_episode(greedy_policy))
        dqn_scores.append(run_episode(dqn_policy))

    g_avg = sum(greedy_scores)/len(greedy_scores)
    d_avg = sum(dqn_scores)/len(dqn_scores)

    # update leaderboard
    leaderboard.append({
        "difficulty": env.difficulty,
        "greedy": round(g_avg,2),
        "dqn": round(d_avg,2)
    })

    df = pd.DataFrame(leaderboard)

    fig = plt.figure()
    plt.plot(greedy_scores, label="Greedy")
    plt.plot(dqn_scores, label="DQN")
    plt.legend()
    plt.title("Performance Comparison")

    return {
        "Greedy Avg": g_avg,
        "DQN Avg": d_avg
    }, fig, df

# ----------------------
# UI
# ----------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 RL Task Scheduler (Advanced)")

    output = gr.JSON()
    graph = gr.Plot()
    table = gr.Dataframe()

    # Difficulty selector
    difficulty = gr.Radio(
        ["easy", "medium", "hard"],
        value="easy",
        label="Select Difficulty"
    )

    difficulty.change(
        fn=set_difficulty,
        inputs=difficulty,
        outputs=[output, graph]
    )

    gr.Button("🔄 Reset").click(
        fn=reset,
        outputs=[output, graph]
    )

    action = gr.Slider(0, 3, step=1, label="Action")

    gr.Button("➡️ Step").click(
        fn=step,
        inputs=action,
        outputs=[output, graph]
    )

    gr.Button("🧠 Train Q").click(
        fn=train_q,
        outputs=[output, graph]
    )

    gr.Button("🔥 Train DQN (Live)").click(
        fn=train_dqn_realtime,
        outputs=[output, graph]
    )

    gr.Button("📊 Evaluate + Leaderboard").click(
        fn=evaluate,
        outputs=[output, graph, table]
    )

demo.launch()
