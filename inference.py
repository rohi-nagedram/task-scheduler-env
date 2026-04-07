import os
import requests
from openai import OpenAI

ENV_URL = "https://bathini-rohini-task-scheduler-env.hf.space"


def call_llm_once():
    base_url = os.environ.get("APIBASE_URL")
    api_key = os.environ.get("HF_TOKEN")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

    print(f"[DEBUG] APIBASE_URL exists: {bool(base_url)}", flush=True)
    print(f"[DEBUG] HF_TOKEN exists: {bool(api_key)}", flush=True)
    print(f"[DEBUG] MODEL_NAME: {model_name}", flush=True)

    if not base_url or not api_key:
        raise Exception("Missing APIBASE_URL or HF_TOKEN")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Return only one number."},
            {"role": "user", "content": "1"}
        ],
        max_tokens=5
    )

    result = response.choices[0].message.content
    print(f"[DEBUG] LLM response: {result}", flush=True)


def main():
    print("[START] task=task-scheduler", flush=True)

    call_llm_once()

    total_reward = 0
    steps = 0

    res = requests.post(f"{ENV_URL}/reset").json()
    state = res["state"]

    for _ in range(20):
        action = state[:4].index(max(state[:4]))

        res = requests.post(
            f"{ENV_URL}/step",
            params={"action": action}
        ).json()

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
