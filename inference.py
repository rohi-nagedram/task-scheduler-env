import os
import requests
import time

# Optional OpenAI (safe usage)
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

try:
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        client = None
except:
    client = None

# Required env variables
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://bathini-rohini-task-scheduler-env.hf.space"
)

MODEL_NAME = os.getenv("MODEL_NAME", "task-scheduler-agent")


# -------------------------------
# SAFE LLM ACTION (WITH FALLBACK)
# -------------------------------
def get_action_from_llm(state):
    if client is None:
        return 0  # fallback immediately

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a task scheduler agent. Return only a number 0 to 3."},
                {"role": "user", "content": f"State: {state}. Choose best action (0-3)."}
            ],
            max_tokens=5
        )

        text = response.choices[0].message.content.strip()
        action = int(text)

        if action < 0 or action > 3:
            return 0

        return action

    except Exception:
        return 0  # CRITICAL fallback


# -------------------------------
# MAIN RUN LOOP
# -------------------------------
def run():
    print("START")

    try:
        # RESET
        res = requests.post(f"{API_BASE_URL}/reset")
        res.raise_for_status()
        data = res.json()

        state = data["state"]
        total_reward = 0

        # EPISODE LOOP
        for step in range(20):

            # choose action
            if USE_OPENAI:
                action = get_action_from_llm(state)
            else:
                action = step % 4  # deterministic fallback policy

            # STEP
            res = requests.post(
                f"{API_BASE_URL}/step",
                params={"action": action}
            )
            res.raise_for_status()

            data = res.json()

            state = data["state"]
            reward = data["reward"]
            done = data["done"]

            total_reward += reward

            # REQUIRED LOG FORMAT
            print(f"STEP {step} | action={action} | reward={reward} | done={done}")

            if done:
                break

            time.sleep(0.1)

        print(f"END total_reward={total_reward}")

    except Exception as e:
        print(f"ERROR: {str(e)}")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run()
