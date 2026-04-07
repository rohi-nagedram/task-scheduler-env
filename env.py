import random

class TaskEnv:
    def __init__(self, difficulty="easy"):
        self.max_tasks = 4
        self.time_budget = 10
        self.difficulty = difficulty
        self.reset()

    def _init_tasks(self):
        def gen():
            return {
                "priority": random.randint(1, 5),
                "time": random.randint(1, 5),
                "deadline": random.randint(3, 10)
            }

        n = {"easy": 2, "medium": 3, "hard": 4}[self.difficulty]
        return [gen() for _ in range(n)]

    def reset(self):
        tasks = self._init_tasks()
        self.active_tasks = len(tasks)
        padded = tasks + [{"priority":0,"time":0,"deadline":0}]*(self.max_tasks-len(tasks))
        self.tasks = padded[:4]
        self.time_left = self.time_budget
        return self.state()

    def state(self):
        vec = [self.time_left]
        for t in self.tasks:
            vec += [t["priority"], t["time"]]
        return vec

    def step(self, action):
        if action >= self.active_tasks:
            return self.state(), -5, False

        task = self.tasks[action]
        self.tasks.pop(action)
        self.tasks.append({"priority":0,"time":0,"deadline":0})
        self.active_tasks -= 1

        self.time_left -= task["time"]

        if self.time_left < 0:
            return self.state(), -10, True

        reward = task["priority"]*2 - task["time"]*0.5
        if self.time_left < task["deadline"]:
            reward -= 2

        done = self.active_tasks == 0 or self.time_left <= 0
        return self.state(), reward, done