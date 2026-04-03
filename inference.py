import os
import requests

# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "dqn")

def run_episode():
    total_reward = 0

    # reset
    res = requests.get(f"{API_BASE_URL}/reset")
    data = res.json()
    state = data["state"]
    done = False

    while not done:
        # simple policy (greedy)
        action = 0
        for i in range(4):
            if state[1 + i*2] != 0:
                action = i
                break

        res = requests.get(f"{API_BASE_URL}/step", params={"action": action})
        data = res.json()

        state = data["state"]
        total_reward += data["reward"]
        done = data["done"]

    return total_reward


def main():
    scores = []

    for _ in range(5):
        score = run_episode()
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print("Scores:", scores)
    print("Average Score:", avg_score)


if __name__ == "__main__":
    main()
