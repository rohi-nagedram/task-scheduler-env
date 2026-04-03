from env import TaskEnv
from agent import choose_action, update_q
from dqn_agent import DQNAgent
from grader import grade_easy, grade_medium, grade_hard

def run_q_learning(level):
    env = TaskEnv(level)

    for _ in range(50):
        state = env.reset()
        while True:
            action = choose_action(state, env.active_tasks, 0.3)
            next_state, reward, done = env.step(action)
            update_q(state, action, reward, next_state, env.active_tasks)
            state = next_state
            if done:
                break

    state = env.reset()
    total = 0
    while True:
        action = choose_action(state, env.active_tasks, 0)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    return total

def run_dqn(level):
    agent = DQNAgent()

    for _ in range(50):
        env = TaskEnv(level)
        state = env.reset()
        while True:
            action = agent.choose_action(state, env.active_tasks, 0.3)
            next_state, reward, done = env.step(action)
            agent.train_step(state, action, reward, next_state, done, env.active_tasks)
            state = next_state
            if done:
                break

    env = TaskEnv(level)
    state = env.reset()
    total = 0
    while True:
        action = agent.choose_action(state, env.active_tasks, 0)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    return total

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        q_score = run_q_learning(level)
        dqn_score = run_dqn(level)

        if level == "easy":
            grade = grade_easy(q_score)
        elif level == "medium":
            grade = grade_medium(q_score)
        else:
            grade = grade_hard(q_score)

        print(f"{level} | Q: {q_score:.2f} | DQN: {dqn_score:.2f} | Grade: {grade:.2f}")