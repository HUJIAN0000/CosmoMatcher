#!/usr/bin/env python3
"""Regenerate manuscript Figures 5 and 6 from the completed stress CSV."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'stress_results')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'paper_figures')
args=parser.parse_args()
output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
summary=pd.read_csv(args.results/'metrics_summary.csv')
common=summary[summary.cohort=='common_all']
baseline=common[(common.tolerance==.05)&(common.truth_model=='flat_LCDM_0p30')].set_index('method')
names=['single','gls_constant','gls_linear']
labels={'single':'Single SN','gls_constant':'Constant GLS','gls_linear':'Linear GLS'}
colors=['#315a86','#ba6c24','#327c68']
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig,axes=plt.subplots(1,3,figsize=(8.3,3.05),layout='constrained')
x=np.arange(3); d=baseline.loc[names]
axes[0].bar(x,1e3*d.bias,color=colors,yerr=1.96e3*d.bias_mcse,capsize=3)
axes[0].axhline(0,color='black',lw=.7); axes[0].set_ylabel(r'Mean signed bias $\times 10^3$')
axes[1].bar(x,d.rmse,color=colors,yerr=1.96*d.rmse_mcse_delta_method,capsize=3)
axes[1].set_ylabel('Pooled RMSE in R')
for shift,key,nominal,hatch in [(-.17,'coverage68_log',68,''),(.17,'coverage95_log',95,'//')]:
    axes[2].bar(x+shift,100*d[key],width=.32,color=colors,hatch=hatch,
                yerr=196*d[key+'_mcse'],capsize=2,edgecolor='white',label=f'{nominal}%')
    axes[2].axhline(nominal,color='gray',ls='--',lw=.8)
axes[2].set_ylabel('Coverage (%)'); axes[2].set_ylim(0,108); axes[2].legend(fontsize=8,loc='lower left')
for i,ax in enumerate(axes):
    ax.set_xticks(x,['Single SN','Constant\nGLS','Linear\nGLS']); ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.set_title(f'({chr(97+i)})',loc='left',fontsize=10)
fig.savefig(output/'mc_benchmark.pdf',bbox_inches='tight')
fig.savefig(output/'mc_benchmark.png',dpi=170,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8.3,3.25),layout='constrained')
for method,color in zip(names,colors):
    allm=common[common.method==method]
    mid=allm[allm.truth_model=='flat_LCDM_0p30'].sort_values('tolerance')
    tol=mid.tolerance.to_numpy()
    ranges=allm.groupby('tolerance').rmse_analytic.agg(['min','max']).reindex(tol)
    axes[0].plot(tol,mid.rmse_analytic,'o-',color=color,label=labels[method],ms=4)
    axes[0].fill_between(tol,ranges['min'],ranges['max'],color=color,alpha=.18)
    for key,style in [('coverage68_log','-'),('coverage95_log','--')]:
        ranges=allm.groupby('tolerance')[key].agg(['min','max']).reindex(tol)
        axes[1].plot(tol,100*mid[key],marker='o',ls=style,color=color,ms=3)
        axes[1].fill_between(tol,100*ranges['min'],100*ranges['max'],color=color,alpha=.18)
axes[0].set_yscale('log'); axes[0].set_ylabel('Analytic pooled RMSE in R');axes[0].legend(fontsize=8)
axes[1].axhline(68,color='gray',lw=.7); axes[1].axhline(95,color='gray',lw=.7)
axes[1].set_ylabel('Coverage (%)');axes[1].set_ylim(40,100)
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([],[],color='black',ls='-',label='68% interval'),Line2D([],[],color='black',ls='--',label='95% interval')],fontsize=8,loc='lower left')
for i,ax in enumerate(axes):
    ax.set_xticks([.03,.05,.07],['3%','5%','7%']);ax.set_xlabel('Relative-distance tolerance')
    ax.grid(alpha=.18);ax.set_title(f'({chr(97+i)})',loc='left',fontsize=10)
fig.savefig(output/'mc_stress.pdf',bbox_inches='tight')
fig.savefig(output/'mc_stress.png',dpi=170,bbox_inches='tight');plt.close(fig)

print(f'Figures saved to {output}')
