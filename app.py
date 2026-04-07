from fastapi import FastAPI
from env import TaskEnv

app = FastAPI()

env = TaskEnv()

# ----------------------
# ROOT (optional)
# ----------------------
@app.get("/")
def home():
    return {
        "message": "Task Scheduler API running",
        "endpoints": ["/reset", "/step", "/state"]
    }

# ----------------------
# RESET (POST REQUIRED)
# ----------------------
@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "state": state,
        "reward": 0,
        "done": False
    }

# ----------------------
# STEP (POST REQUIRED)
# ----------------------
@app.post("/step")
def step(action: int):
    state, reward, done = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }

# ----------------------
# STATE (GET OK)
# ----------------------
@app.get("/state")
def get_state():
    return env.state()
