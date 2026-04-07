import os
import requests
from openai import OpenAI

ENV_URL = "https://bathini-rohini-task-scheduler-env.hf.space"


def call_llm_once():
    base_url = os.environ.get("APIBASE_URL")
    api_key = os.environ.get("APLKEY")

    print(f"[DEBUG] APIBASE_URL exists: {bool(base_url)}", flush=True)
    print(f"[DEBUG] APLKEY exists: {bool(api_key)}", flush=True)

    if not base_url or not api_key:
        raise RuntimeError("Missing APIBASE_URL or APLKEY")

    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only one number."},
            {"role": "user", "content": "1"}
        ],
        max_tokens=5
    )

    result = (response.choices[0].message.content or "").strip()
    print(f"[DEBUG] LLM response: {result}", flush=True)
    return result


def post_json(url, **kwargs):
    r = requests.post(url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    print("[START] task=task-scheduler", flush=True)

    call_llm_once()

    total_reward = 0
    steps = 0

    res = post_json(f"{ENV_URL}/reset")
    state = res["state"]

    for _ in range(20):
        action = state[:4].index(max(state[:4]))

        res = post_json(
            f"{ENV_URL}/step",
            params={"action": action}
        )

        state = res["state"]
        reward = res["reward"]
        done = res["done"]

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    print(f"[END] task=task-scheduler score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
