import requests
import random

BASE_URL = "https://bathini-rohini-task-scheduler-env.hf.space"

# -----------------------------
# Q-TABLE
# -----------------------------
Q = {}

def get_state_key(state):
    return tuple(state[:4])

def choose_action(state, epsilon=0.1):
    key = get_state_key(state)

    if key not in Q:
        Q[key] = [0.0, 0.0, 0.0, 0.0]

    # calculate value = priority - cost idea
    values = state[:4]

    # avoid picking same top repeatedly
    sorted_actions = sorted(range(4), key=lambda i: values[i], reverse=True)

    # small exploration
    if random.random() < epsilon:
        return random.choice(sorted_actions[:2])  # top 2 only

    # Q-learning preference
    q_best = Q[key].index(max(Q[key]))

    return q_best if max(Q[key]) > 0 else sorted_actions[0]


def update_q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    key = get_state_key(state)
    next_key = get_state_key(next_state)

    if next_key not in Q:
        Q[next_key] = [0.0, 0.0, 0.0, 0.0]

    old_value = Q[key][action]
    next_max = max(Q[next_key])

    # Q-learning update
    Q[key][action] = old_value + alpha * (reward + gamma * next_max - old_value)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("[START] task=task-scheduler", flush=True)

    total_reward = 0
    steps = 0

    # RESET
    res = requests.post(f"{BASE_URL}/reset").json()
    state = res["state"]

    for step in range(20):
        action = choose_action(state)

        res = requests.post(
            f"{BASE_URL}/step",
            params={"action": action}
        ).json()

        next_state = res["state"]
        reward = res["reward"]

        done = res["done"]

        update_q(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    print(f"[END] task=task-scheduler score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
