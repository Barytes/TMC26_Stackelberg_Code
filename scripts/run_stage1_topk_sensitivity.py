#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.baselines import run_stage2_solver
from tmc26_exp.stackelberg import _build_data, _candidate_family, _provider_revenue, _candidate_revenue_estimate

def topk_gain(users, offloading_set, pE, pN, provider, system, k):
    data=_build_data(users)
    X=tuple(sorted(offloading_set))
    cur=_provider_revenue(data, X, pE, pN, provider, system)
    fam=_candidate_family(data, X, pE, pN, system)
    scored=[]
    for Y in fam:
        s=_candidate_revenue_estimate(data, users, Y, pE, pN, provider, system, estimator_variant='boundary')
        if s is not None: scored.append((float(s),Y))
    scored.sort(key=lambda t:t[0], reverse=True)
    best=0.0
    for _,Y in scored[:min(k,len(scored))]:
        rev=_provider_revenue(data,Y,pE,pN,provider,system)
        best=max(best,float(rev-cur))
    return best

def argmin(z,pE,pN):
    j,i=np.unravel_index(np.argmin(z),z.shape)
    return i,j,float(pE[i]),float(pN[j]),float(z[j,i])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/stage1_fig6_extdomain.toml')
    ap.add_argument('--n-users',type=int,default=40)
    ap.add_argument('--seed',type=int,default=20260002)
    ap.add_argument('--grid-points',type=int,default=21)
    ap.add_argument('--k-list',default='2,4,8,16,32')
    ap.add_argument('--run-name',default='')
    args=ap.parse_args()
    cfg=load_config(args.config)
    cfg=type(cfg)(**{**cfg.__dict__,'n_users':args.n_users})
    pE=np.linspace(cfg.system.cE,cfg.baselines.max_price_E,args.grid_points)
    pN=np.linspace(cfg.system.cN,cfg.baselines.max_price_N,args.grid_points)
    out=Path(cfg.output_dir)/(args.run_name or f'stage1_topk_sensitivity_{int(time.time())}')
    out.mkdir(parents=True,exist_ok=True)
    users=sample_users(cfg,np.random.default_rng(args.seed))
    shape=(len(pN),len(pE))
    esp=np.zeros(shape); nsp=np.zeros(shape)
    sets=[[None for _ in pE] for __ in pN]
    for j,pn in enumerate(pN):
        for i,pe in enumerate(pE):
            r=run_stage2_solver(cfg.baselines.stage2_solver_for_pricing, users, float(pe), float(pn), cfg.system, cfg.stackelberg, cfg.baselines)
            esp[j,i]=r.esp_revenue; nsp[j,i]=r.nsp_revenue; sets[j][i]=r.offloading_set
    eps_real=np.maximum(np.max(esp,axis=1,keepdims=True)-esp, np.max(nsp,axis=0,keepdims=True)-nsp)
    i0,j0,p0e,p0n,v0=argmin(eps_real,pE,pN)
    rows=[]
    k_list=[int(x) for x in args.k_list.split(',') if x.strip()]
    for k in k_list:
        t0=time.perf_counter()
        eh=np.zeros(shape)
        for j,pn in enumerate(pN):
            for i,pe in enumerate(pE):
                X=sets[j][i]
                gE=topk_gain(users,X,float(pe),float(pn),'E',cfg.system,k)
                gN=topk_gain(users,X,float(pe),float(pn),'N',cfg.system,k)
                eh[j,i]=max(gE,gN)
        dt=time.perf_counter()-t0
        diff=eh-eps_real
        i,j,ph,pn,val=argmin(eh,pE,pN)
        rows.append({
            'k':k,'mae':float(np.mean(np.abs(diff))),'rmse':float(np.sqrt(np.mean(diff*diff))),
            'max_abs_diff':float(np.max(np.abs(diff))),'argmin_hat_pE':ph,'argmin_hat_pN':pn,
            'argmin_real_pE':p0e,'argmin_real_pN':p0n,
            'argmin_price_distance':float(np.hypot(ph-p0e,pn-p0n)),'runtime_sec':dt
        })
    with (out/'summary_topk_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ks=[r['k'] for r in rows]
    fig,axs=plt.subplots(1,3,figsize=(13,4),dpi=160)
    axs[0].plot(ks,[r['mae'] for r in rows],'-o',label='MAE'); axs[0].plot(ks,[r['rmse'] for r in rows],'-s',label='RMSE'); axs[0].set_title('Error vs K'); axs[0].set_xlabel('K'); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(ks,[r['argmin_price_distance'] for r in rows],'-o',c='tab:red'); axs[1].set_title('Argmin distance vs K'); axs[1].set_xlabel('K'); axs[1].grid(alpha=.3)
    axs[2].plot(ks,[r['runtime_sec'] for r in rows],'-o',c='tab:green'); axs[2].set_title('Runtime vs K'); axs[2].set_xlabel('K'); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'k_sensitivity.png'); plt.close(fig)
    print('done',out)

if __name__ == '__main__':
    main()
