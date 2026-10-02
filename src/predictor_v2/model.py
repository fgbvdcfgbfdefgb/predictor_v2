import torch, torch.nn as nn
class ActorCritic(nn.Module):
 def __init__(self,n,hidden=256,actions=7):
  super().__init__(); self.body=nn.Sequential(nn.Linear(n,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh()); self.actor=nn.Linear(hidden,actions); self.critic=nn.Linear(hidden,1)
 def forward(self,x): h=self.body(x); return self.actor(h),self.critic(h).squeeze(-1)
