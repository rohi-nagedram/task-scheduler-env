import os
import requests
from openai import OpenAI

# ----------------------
# ENV VARIABLES
# ----------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://bathini-rohini-task-scheduler-env.hf.space"
)

MODEL_NAME = os.getenv("MODEL_NAME", "dqn")  # dqn | greedy | llm
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------
# OPENAI CLIENT (SAFE INIT)
# ----------------------

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------
# POLICIES
# ----------------------

def greedy_policy(state):
    for i in range(4):
        if state[1 + i * 2] != 0:
            return i
    return 0


def llm_policy(state):
    if client is None:
        return greedy_policy(state)

    try:
        prompt = f"""
You are a task scheduling agent.

State format:
[time_left, p1, t1, p2, t2, p3, t3, p4, t4]

State:
{state}

Choose best action index (0-3).
Return ONLY a number.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )

        action = int(response.choices[0].message.content.strip())
        return max(0, min(3, action))

    except:
        return greedy_policy(state)


def select_action(state):
    if MODEL_NAME == "llm":
        return llm_policy(state)
    else:
        return greedy_policy(state)


# ----------------------
# RUN EPISODE
# ----------------------

def run_episode():
    print("START")

    total_reward = 0

    # RESET
    res = requests.get(f"{API_BASE_URL}/reset")
    data = res.json()
    state = data["state"]
    done = False

    while not done:
        print("STEP")

        action = select_action(state)

        res = requests.get(
            f"{API_BASE_URL}/step",
            params={"action": action}
        )
        data = res.json()

        state = data["state"]
        reward = data["reward"]
        done = data["done"]

        total_reward += reward

    print("END")

    return total_reward


# ----------------------
# MAIN
# ----------------------

def main():
    scores = []

    for _ in range(5):
        score = run_episode()
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print("FINAL_SCORES:", scores)
    print("AVERAGE_SCORE:", avg_score)


if __name__ == "__main__":
    main()
