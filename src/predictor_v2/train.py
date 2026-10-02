import argparse, os, random, json, yaml, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from .data import discover,load_day,features
from .analyser import MarketAnalyser
from .model import ActorCritic

def setup(seed):
 rank=int(os.getenv('RANK',0)); world=int(os.getenv('WORLD_SIZE',1)); local=int(os.getenv('LOCAL_RANK',0))
 if world>1: torch.distributed.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
 random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
 dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu');
 if dev.type=='cuda': torch.cuda.set_device(dev)
 return rank,world,dev

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/default.yaml'); a=ap.parse_args(); c=yaml.safe_load(open(a.config)); rank,world,dev=setup(c['seed']); days=discover(c['data_root'])
 if not days: raise SystemExit('No complete days found. Expected data/processed/<exchange>/<symbol>/YYYY-MM-DD.parquet')
 sample=load_day(next(iter(days.values())),c['symbols']); nf=features(sample,c['symbols']).shape[1]+1
 net=ActorCritic(nf,c['ppo']['hidden']).to(dev); model=DDP(net,device_ids=[dev.index]) if world>1 and dev.type=='cuda' else net; opt=torch.optim.Adam(model.parameters(),lr=c['ppo']['lr']); analyser=MarketAnalyser(c['analyser_trees'],c['seed']); out=Path(c['artifacts']); out.mkdir(exist_ok=True)
 history=[]
 for ep in range(c['epochs']):
  day=random.choice(sorted(days)); df=load_day(days[day],c['symbols']); F=features(df,c['symbols']); close=df[f"{c['symbols'][0]}_close"]
  analyser.fit(F.to_numpy(),close); pred,regime=analyser.feedback(F.to_numpy()); X=np.c_[F.to_numpy(),pred].astype('float32'); rets=np.nan_to_num(close.pct_change().to_numpy(),nan=0)
  actions=[]; logps=[]; vals=[]; rewards=[]; equities=[c['initial_cash']]; prev=0
  for t in range(min(len(X)-1,c['steps_per_epoch'])):
   xt=torch.from_numpy(X[t]).to(dev); logits,val=model(xt); dist=Categorical(logits=logits); act=dist.sample(); pos=int(act.item())-3; cost=abs(pos-prev)*(c['fee_bps']+c['slippage_bps'])/1e4; rew=pos*rets[t+1]-cost+c['ppo']['adviser_reward_weight']*np.sign(pos)*pred[t]
   actions.append(act); logps.append(dist.log_prob(act)); vals.append(val); rewards.append(float(rew)); equities.append(equities[-1]*(1+rew)); prev=pos
  R=0; returns=[]
  for r in rewards[::-1]: R=r+c['ppo']['gamma']*R; returns.append(R)
  returns=torch.tensor(returns[::-1],device=dev); values=torch.stack(vals); old=torch.stack(logps).detach(); adv=(returns-values.detach()); adv=(adv-adv.mean())/(adv.std()+1e-8)
  idx=torch.randperm(len(rewards),device=dev)
  for _ in range(c['ppo']['update_epochs']):
   for b in idx.split(c['ppo']['batch_size']):
    logits,v=model(torch.from_numpy(X[:len(rewards)]).to(dev)[b]); d=Categorical(logits=logits); lp=d.log_prob(torch.stack(actions)[b]); ratio=(lp-old[b]).exp(); s1=ratio*adv[b]; s2=ratio.clamp(1-c['ppo']['clip'],1+c['ppo']['clip'])*adv[b]; loss=-torch.min(s1,s2).mean()+.5*(v-returns[b]).pow(2).mean()-c['ppo']['entropy_coef']*d.entropy().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
  advice=analyser.advice(pred,regime); metric={"epoch":ep,"day":day,"return":equities[-1]/equities[0]-1,"reward":sum(rewards),**advice}; history.append(metric)
  if rank==0:
   fig,ax=plt.subplots(2,1,figsize=(12,7)); ax[0].plot(equities); ax[0].set_title(f'Epoch {ep} | {day} | equity'); ax[1].plot(np.array(actions)-3,alpha=.7); ax[1].set_title('position (-3..3)'); fig.suptitle(json.dumps(advice)); fig.tight_layout(); fig.savefig(out/f'epoch_{ep:05d}.png',dpi=130); plt.close(fig); json.dump(metric,open(out/f'epoch_{ep:05d}_advisor.json','w'),indent=2); json.dump(history,open(out/'metrics.json','w'),indent=2)
   if ep%c['checkpoint_every']==0: torch.save((model.module if hasattr(model,'module') else model).state_dict(),out/f'policy_{ep:05d}.pt')
 if world>1: torch.distributed.destroy_process_group()
if __name__=='__main__': main()
