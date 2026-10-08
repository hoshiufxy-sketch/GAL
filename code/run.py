"""Prospectively fixed computational diagnostic, not a prospective wet-lab study."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,itertools,argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score,roc_auc_score
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'unified_top20_guided_round1_20260922/source_snapshot'))
from gal.features import extract_complete_features
from ensemble_learner import XGB_PARAMS
METHODS=['regression','classifier','rank','joint']
MEMBERS=16

def features(d):
    seq=d.spacer.tolist();L=np.array(list(map(len,seq)))
    km=[''.join(p) for k in (1,2,3) for p in itertools.product('ACGT',repeat=k)]
    phys=d.spacer.apply(extract_complete_features)[['Total_MW','Total_Stacking','GC_Front','GC_Back','DG_total']].to_numpy()
    X=np.column_stack([L==6,L==7,L==8,[[s.count(b)/len(s) for b in 'ACGT'] for s in seq],
        [[sum(s[j:j+len(k)]==k for j in range(len(s)-len(k)+1))/(len(s)-len(k)+1) for k in km] for s in seq],
        [[int(j<len(s) and s[j]==b) for j in range(8) for b in 'ACGT'] for s in seq],phys])
    assert X.shape==(len(d),128) and np.isfinite(X).all() and d.spacer.is_unique
    return X

def objective(kind):
    # n * [mean BCE + mean pair loss]/2 for joint. Diagonal Hessian upper bound
    # 2*degree bounds the positive semidefinite pairwise graph Laplacian.
    def obj(y,s):
        n=len(y);p=expit(s)
        g=p-y if kind!='rank' else np.zeros(n)
        h=p*(1-p) if kind!='rank' else np.zeros(n)
        if kind!='classifier':
            ip=np.flatnonzero(y==1);ineg=np.flatnonzero(y==0)
            if len(ip) and len(ineg):
                q=expit(s[ineg][None,:]-s[ip][:,None]);c=n/q.size
                g[ip]-=c*q.sum(1);g[ineg]+=c*q.sum(0)
                v=2*c*q*(1-q);h[ip]+=v.sum(1);h[ineg]+=v.sum(0)
            if kind=='joint':g=g/2;h=h/2
        return g,np.maximum(h,1e-6)
    return obj

def loss(kind,y,s):
    b=np.mean(np.logaddexp(0,s)-y*s)
    pair=np.mean(np.logaddexp(0,s[y==0][None,:]-s[y==1][:,None]))
    return len(y)*(b if kind=='classifier' else pair if kind=='rank' else (b+pair)/2)

def unit_checks():
    y=np.array([0.,1.,0.,1.,0.]);s=np.array([-.2,.3,.1,-.4,.5]);eps=1e-5
    for kind in METHODS[1:]:
        g,h=objective(kind)(y,s)
        numeric=np.array([(loss(kind,y,s+np.eye(5)[i]*eps)-loss(kind,y,s-np.eye(5)[i]*eps))/(2*eps) for i in range(5)])
        assert np.allclose(g,numeric,atol=1e-7),(kind,g,numeric)
        H=np.column_stack([(objective(kind)(y,s+np.eye(5)[i]*eps)[0]-objective(kind)(y,s-np.eye(5)[i]*eps)[0])/(2*eps) for i in range(5)])
        assert np.linalg.eigvalsh(np.diag(h)-H).min()>-1e-7

def predict(X,y,Z,kind,tau,seed,params=None,members=MEMBERS):
    # This function receives ONLY revealed training labels.
    rng=np.random.default_rng(seed);out=[]
    target=np.log1p(y) if kind=='regression' else (y>=tau).astype(float)
    center=target.mean() if kind=='regression' else 0
    scale=target.std() if kind=='regression' else 1
    for m in range(members):
        boot=rng.integers(0,len(y),len(y))
        p=dict(XGB_PARAMS if params is None else params,n_jobs=1,random_state=seed+m,verbosity=0,base_score=0.)
        if kind!='regression':p['objective']=objective(kind)
        model=XGBRegressor(**p)
        model.fit(X[boot],((target-center)/scale)[boot])
        out.append(model.predict(Z)*scale+center)
    a=np.asarray(out)
    return a.mean(0),a.std(0,ddof=1)

def score(mu,sd,policy):
    if policy=='sigma':return sd
    if policy in ['greedy','static']:return mu
    return (mu-mu.mean())/max(mu.std(),1e-12)+.35*(sd-sd.mean())/max(sd.std(),1e-12)

def cv(d,X,name):
    y=d.norm.to_numpy();L=d.spacer.str.len().to_numpy();rows=[];pred=[]
    for rep in range(3):
        for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=9243100+rep).split(X,L)):
            tau=np.quantile(y[tr],.8);high=y[te]>=tau;k=int(np.ceil(.1*len(te)))
            for kind in METHODS:
                mu,sd=predict(X[tr],y[tr],X[te],kind,tau,9244000+100*rep+f)
                chosen=np.argsort(-mu,kind='stable')[:k];hits=int(high[chosen].sum())
                rows.append(dict(dataset=name,repeat=rep,fold=f,model=kind,n=len(te),k=k,hits=hits,expected=k*high.mean(),precision=hits/k,ap=average_precision_score(high,mu),auc=roc_auc_score(high,mu),threshold=tau))
                for j,i in enumerate(te):pred.append(dict(dataset=name,repeat=rep,fold=f,model=kind,spacer=d.spacer.iloc[i],observed=y[i],threshold=tau,score=mu[j],sd=sd[j],selected=int(j in chosen)))
        print('CV',name,rep,flush=True)
    pd.DataFrame(rows).to_csv(R/f'cv_{name}.csv',index=False)
    pd.DataFrame(pred).to_csv(R/f'predictions_{name}.csv',index=False)

def replay(d,X,name,trials):
    y=d.norm.to_numpy();n=len(y);allidx=np.arange(n)
    # Full-pool labels are used ONLY in the evaluator, never training or acquisition.
    top=set(np.argsort(-y,kind='stable')[:20]);rows=[];trace=[]
    arms=[(m,p) for m in METHODS for p in ['greedy','ucb']]+[('regression','sigma'),('regression','static'),('none','random')]
    for trial in range(trials):
        initial=np.random.default_rng(9245000+trial).choice(n,48,replace=False)
        tau=np.quantile(y[initial],.8)
        for kind,policy in arms:
            known=initial.copy();static=None
            for stage in range(4):
                batch=initial if stage==0 else chosen
                rows.append(dict(dataset=name,trial=trial,model=kind,policy=policy,stage=stage,n=len(known),threshold=tau,top20=sum(i in top for i in known),high=int((y[known]>=tau).sum()),batch_hits=int((y[batch]>=tau).sum()),batch_size=len(batch),batch_mean=y[batch].mean(),best=y[known].max()))
                for i in batch:trace.append(dict(dataset=name,trial=trial,model=kind,policy=policy,stage=stage,index=int(i),spacer=d.spacer.iloc[i]))
                if stage==3:break
                pool=np.setdiff1d(allidx,known)
                if policy=='random':
                    chosen=np.random.default_rng(9246000+trial*10+stage).choice(pool,40,replace=False)
                else:
                    if policy=='static' and static is not None:mu,sd=static[0][pool],static[1][pool]
                    else:
                        mu,sd=predict(X[known],y[known],X if policy=='static' else X[pool],kind,tau,9247000+100*trial+stage)
                        if policy=='static':static=(mu,sd);mu,sd=mu[pool],sd[pool]
                    chosen=pool[np.lexsort((pool,-score(mu,sd,policy)))[:40]]
                assert not np.intersect1d(chosen,known).size and len(np.unique(chosen))==40
                known=np.r_[known,chosen]
        pd.DataFrame(rows).to_csv(R/f'replay_{name}.csv',index=False)
        pd.DataFrame(trace).to_csv(R/f'trace_{name}.csv',index=False)
        print('REPLAY',name,trial,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cv','replay','check'],default='check');ap.add_argument('--dataset',default='484');ap.add_argument('--trials',type=int,default=20);a=ap.parse_args()
    unit_checks()
    sources={'484':B/'unified_top20_guided_round1_20260922/data_484.csv','421':B/'data_recheck_20260922/sequence_means.csv'}
    protocol=dict(sources={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in sources.items()},features='mixed128',members=MEMBERS,params=XGB_PARAMS,threshold='training 80th percentile; replay initial48 threshold fixed for all rounds',joint_loss='0.5 * (mean BCE + mean logistic positive-negative pair loss)',cv='3 repeats x 5 folds stratified by length; top ceil(10% heldout) precision',replay='20 paired trials 48+40+40+40; all four objectives crossed with greedy and z(mu)+0.35*z(sd)',calibration='classification/rank margins are scores, not calibrated probabilities',status='Exploratory fixed configurations; no claim of untouched external data or nested tuning',limitations=['421 exclusions lack finalized QC','484 low OD and repeat dispersion retained','same sequence aggregated before splitting','no batch/homology-separated validation','bootstrap SD not calibrated posterior uncertainty'])
    (R/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    if a.mode=='check':print('Gradient and Hessian checks passed');return
    d=pd.read_csv(sources[a.dataset]);X=features(d)
    with threadpool_limits(limits=1):
        if a.mode=='cv':cv(d,X,a.dataset)
        else:replay(d,X,a.dataset,a.trials)

if __name__=='__main__':main()
