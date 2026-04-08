import os
import requests

ENV_URL = "https://bathini-rohini-task-scheduler-env.hf.space"


# -----------------------------
# LLM VIA PROXY (DIRECT CALL)
# -----------------------------
def get_action_from_llm(state):
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    if not base_url or not api_key:
        raise Exception("Missing API_BASE_URL or API_KEY")

    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Select best task index (0-3)."},
            {"role": "user", "content": f"State: {state}. Return only a number."}
        ],
        "max_tokens": 5
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()

    try:
        action = int(text)
    except:
        action = 0

    return max(0, min(3, action))


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("[START] task=task-scheduler", flush=True)
    print("VERSION: FINAL_DIRECT_PROXY", flush=True)

    total_reward = 0
    steps = 0

    # RESET ENV
    res = requests.post(f"{ENV_URL}/reset")
    res.raise_for_status()
    state = res.json()["state"]

    # LOOP
    for _ in range(20):
        # 🔥 LLM decides action EVERY STEP
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
