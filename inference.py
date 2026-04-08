import os
import requests
from openai import OpenAI

# ✅ DO NOT CHANGE THIS
ENV_URL = "https://bathini-rohini-task-scheduler-env.hf.space"


# -----------------------------
# FORCE LLM CALL (DETECTABLE)
# -----------------------------
def get_action_from_llm(state):
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    # ❌ HARD FAIL (so validator sees issue clearly)
    if not base_url or not api_key:
        raise Exception("API_BASE_URL or API_KEY missing")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a task scheduler agent. Choose best action index (0-3)."
            },
            {
                "role": "user",
                "content": f"State: {state}. Return ONLY a number between 0 and 3."
            }
        ],
        max_tokens=5
    )

    text = response.choices[0].message.content.strip()

    try:
        action = int(text)
    except:
        action = 0  # fallback ONLY after LLM call

    return max(0, min(3, action))


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    print("[START] task=task-scheduler", flush=True)
    print("VERSION: FINAL_PROXY_V2", flush=True)  # 🔥 DEPLOYMENT CHECK

    total_reward = 0
    steps = 0

    # RESET ENV
    res = requests.post(f"{ENV_URL}/reset")
    res.raise_for_status()
    state = res.json()["state"]

    # RUN LOOP
    for _ in range(20):
        # 🔥 LLM MUST RUN EVERY STEP
        action = get_action_from_llm(state)

        res = requests.post(
            f"{ENV_URL}/step",
            params={"action": action}
        )
        res.raise_for_status()

        data = res.json()
        state = data["state"]
        reward = data["reward"]
        done = data["done"]

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    print(f"[END] task=task-scheduler score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
