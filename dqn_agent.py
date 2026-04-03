import torch, torch.nn as nn, torch.optim as optim, random

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,4)
        )
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self):
        self.model = Net()
        self.target = Net()
        self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.9
        self.step = 0

    def select_action(self,s):
        if random.random()<0.2: return random.randint(0,3)
        s=torch.tensor(s,dtype=torch.float32)
        return torch.argmax(self.model(s)).item()

    def train_step(self,s,a,r,ns,done):
        s=torch.tensor(s,dtype=torch.float32)
        ns=torch.tensor(ns,dtype=torch.float32)

        q=self.model(s)
        target=q.clone().detach()

        if done:
            target[a]=r
        else:
            target[a]=r+self.gamma*torch.max(self.target(ns))

        loss=nn.MSELoss()(q,target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.step+=1
        if self.step%20==0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self): torch.save(self.model.state_dict(),"dqn.pth")
    def load(self):
        try: self.model.load_state_dict(torch.load("dqn.pth"))
        except: pass
