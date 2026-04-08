import os
import requests
from openai import OpenAI

ENV_URL = "https://bathini-rohini-task-scheduler-env.hf.space"


def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


def call_llm_once():
    raw_base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not raw_base_url or not api_key:
        raise Exception("Missing API_BASE_URL or API_KEY")

    base_url = normalize_base_url(raw_base_url)

    print(f"[DEBUG] API_BASE_URL raw exists: {bool(raw_base_url)}", flush=True)
    print(f"[DEBUG] API_BASE_URL final: {base_url}", flush=True)
    print(f"[DEBUG] API_KEY exists: {bool(api_key)}", flush=True)
    print(f"[DEBUG] MODEL_NAME: {model_name}", flush=True)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Reply with exactly: 1"},
            {"role": "user", "content": "1"}
        ],
        max_tokens=5,
    )

    result = (response.choices[0].message.content or "").strip()
    print(f"[DEBUG] LLM response: {result}", flush=True)
    return result


def main():
    print("[START] task=task-scheduler", flush=True)

    llm_result = call_llm_once()
    print(f"[DEBUG] LLM call completed before env reset/step. result={llm_result}", flush=True)

    total_reward = 0
    steps = 0

    res = requests.post(f"{ENV_URL}/reset", timeout=30)
    res.raise_for_status()
    state = res.json()["state"]

    for _ in range(20):
        action = state[:4].index(max(state[:4]))

        res = requests.post(
            f"{ENV_URL}/step",
            params={"action": action},
            timeout=30
        )
        res.raise_for_status()
        body = res.json()

        state = body["state"]
        reward = body["reward"]
        done = body["done"]

        total_reward += reward
        steps += 1

        print(f"[STEP] step={steps} reward={reward}", flush=True)

        if done:
            break

    print(f"[END] task=task-scheduler score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
