"""Summarize only the two original-program runs; no alternative sampler or model."""
from pathlib import Path
import json,csv,re,hashlib,shutil,importlib.util,sys
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import emcee
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cosmo_tools import plot_getdist_comparison
R=Path(__file__).resolve().parent
NAMES=['alpha','beta','f_e','eta0','gamma','m_step','tau']
CASES=['with_systematics','without_systematics']

def load_model(case):
    spec=importlib.util.spec_from_file_location('model_'+case,R/'runs'/case/'cddv301.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    out=R/'output';out.mkdir(exist_ok=True)
    summaries={};flats={}
    for case in CASES:
        folder=R/'runs'/case;flat=np.load(folder/'flat_samples.npy');chain=np.load(folder/'chain.npy')
        assert chain.shape==(1000,140,7)
        tau=emcee.autocorr.integrated_time(chain,quiet=True)
        burn=int(2*tau.max());thin=max(1,int(.5*tau.min()))
        assert np.array_equal(flat,chain[burn+thin-1::thin].reshape(-1,7))
        q=np.percentile(flat,[2.5,16,50,84,97.5],axis=0)
        log=(folder/'run.log').read_text(encoding='utf-8')
        best=float(re.search(r'Min chi2\s*=\s*([\d.]+)',log)[1])
        s={'case':case,'parameters':{n:q[:,j].tolist() for j,n in enumerate(NAMES)},'percentile_order':[2.5,16,50,84,97.5],
           'walkers':140,'steps':1000,'workers':8,'burn':burn,'thin':thin,'flat_sample_count':len(flat),
           'autocorrelation_times':tau.tolist(),'passes_original_50tau_length_check':bool(1000>=50*tau.max()),
           'mean_acceptance':float(np.load(folder/'acceptance.npy').mean()),'minimum_chi2_optimizer':best,
           'mass_step_sample_range':[float(flat[:,5].min()),float(flat[:,5].max())],
           'seed':'Original initial-position generator seed=42; wrapper fixes numpy/emcee seed=20261019 in both cases.'}
        (folder/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');summaries[case]=s;flats[case]=flat
    full=load_model('with_systematics');stat=load_model('without_systematics')
    assert np.array_equal(np.loadtxt(R/'runs/with_systematics/result_final_min_select_handle'),np.loadtxt(R/'runs/without_systematics/result_final_min_select_handle'))
    for j in range(21):assert np.array_equal(full._DATA[j],stat._DATA[j])
    assert stat._DATA[-1] is None
    assert np.array_equal(full.dM,stat.dM) and np.array_equal(full.dM_s,stat.dM_s)
    theta=np.array([.31,4.7,1.025,.03,.06,10.06,.127])
    diagonal=np.diag(full.dmbl**2+full.dmbs**2)
    diff=abs(full.lnlike(theta,*full._DATA[:-1],diagonal)-stat.lnlike(theta,*stat._DATA))
    assert diff<1e-10
    manifest=json.loads((R/'manifest.json').read_text())
    assert hashlib.sha256((R/'runs/with_systematics/cddv301.py').read_bytes()).hexdigest()==manifest['cddv301.py']['sha256']
    plot_getdist_comparison(
        samples_list=[flats[case] for case in CASES],
        labels=[r'\alpha',r'\beta',r'f_e',r'\eta_0',r'\gamma',r'm_{\rm step}',r'\tau'],
        legend_labels=['With systematic covariance','Without systematic covariance'],
        colors=['#2368ad','#d77e25'],
        filename=str(out/'cddr143_triangle.pdf'),
    )
    plt.savefig(out/'cddr143_triangle.png',dpi=160,bbox_inches='tight')
    plt.close('all')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(10,4.2))
    mass_limits=(min(f[:,5].min() for f in flats.values())-.1,max(f[:,5].max() for f in flats.values())+.1)
    for case,label,color in zip(CASES,['With C11 systematics and cross blocks','Without C11 systematics'],['#2368ad','#d77e25']):
        for ax,j,limits in [(axs[0],3,(-.5,.5)),(axs[1],5,mass_limits)]:
            density,edges=np.histogram(flats[case][:,j],bins=45,range=limits,density=True)
            ax.plot((edges[1:]+edges[:-1])/2,gaussian_filter1d(density,1),color=color,label=label,lw=1.8)
    axs[0].set_xlabel(r'$\eta_0$ (code convention)');axs[0].axvline(0,c='.5',ls='--',lw=1)
    axs[1].set_xlabel(r'$m_{\rm step}$')
    for ax in axs:ax.set_ylabel('Marginal density');ax.spines[['top','right']].set_visible(False)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=8.8,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2)
    fig.suptitle('cddv301.py: same matching, priors and sampling settings',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.83));fig.savefig(out/'cddr143_covariance_comparison.pdf');fig.savefig(out/'cddr143_covariance_comparison.png',dpi=160);plt.close(fig)
    with (out/'parameter_constraints.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['case','parameter','q2.5','q16','q50','q84','q97.5'])
        for case in CASES:
            for name in NAMES:w.writerow([case,name,*summaries[case]['parameters'][name]])
    allsummary={'protocol':'cddv301.py original main block','cases':summaries,'only_model_difference':'_C_DELTA_MU=None in statistical-only case',
       'host_error_99':'unchanged; no special handling','same_99_matched_systems':True,'diagonal_limit_log_kernel_difference':float(diff),
       'triangle_plotter':'cosmo_tools.plot_getdist_comparison',
       'plotting_helper_sha256':hashlib.sha256((R/'cosmo_tools.py').read_bytes()).hexdigest(),
       'triangle_cases':CASES,
       'triangle_sample_sha256':{case:hashlib.sha256((R/'runs'/case/'flat_samples.npy').read_bytes()).hexdigest() for case in CASES},
       'figure_provenance':'Figure 7 overlays the two recorded cddv301 runs using cosmo_tools.py; the table and Figure 8 use the same recorded samples. The original author-provided single-case PDF remains an input reference.'}
    (out/'comparison_summary.json').write_text(json.dumps(allsummary,indent=2),encoding='utf-8')
    rows=[]
    for name in NAMES:
        cells=[]
        for case in CASES:
            q=summaries[case]['parameters'][name];cells.append(f'{q[2]:.4f} +{q[3]-q[2]:.4f} -{q[2]-q[1]:.4f}')
        rows.append('| '+name+' | '+' | '.join(cells)+' |')
    report='''# 按cddv301.py原始流程整理的科学应用

本次仅采用作者要求的固定流程：原始模型与先验、Nelder–Mead优化初值、140个walker、1000步、8进程、emcee默认移动，以及原程序自相关时间决定的burn/thin。未使用额外的全局提议、独立集合拼接或加长采样。包装脚本只固定随机数种子并保存程序实际用于绘图的样本，便于复算。

仅比较两种情况：包含C11系统误差子矩阵（LL、SS、LS及其转置）与不包含这些系统误差。无系统误差版本只在_DATA打包前加入_C_DELTA_MU=None，调用原程序自带的对角误差分支；AST核对确认其余程序逻辑相同。两组都保留相同99组配对、相同统计误差和宿主质量误差，质量误差99既不剔除也不置零。

论文Figure 7使用用户提供的cosmo_tools.py中的plot_getdist_comparison，将两组原程序实际绘图样本叠加：蓝色为含系统误差子矩阵及交叉协方差，橙色为不含系统误差子矩阵。Figure 7、Figure 8和参数表均来自同一对已记录的原流程复跑样本；本次仅更新绘图，没有重新采样或改变参数结果。作者此前提供的单组原图仍保留作输入参考。

## 本次复跑参数区间

下表为中位数及16/84百分位等尾区间。原代码eta0对应DA(1+z)^2/DL=1+eta0*z，保留原图eta0标签，并在正文明确它与Zhou 2021打印定义的方向差异。

| 参数 | 含系统误差及交叉块 | 不含系统误差 |
|---|---|---|
'''+ '\n'.join(rows)+f'''

两组保留绘图样本的m_step范围分别为{summaries['with_systematics']['mass_step_sample_range']}和{summaries['without_systematics']['mass_step_sample_range']}。本次固定流程的样本集中在约10附近，正文不使用此前额外采样所得约6分支、宿主误差置零或归一化似然对照的图表和参数。

## 程序自身的运行诊断

含系统项：acceptance={summaries['with_systematics']['mean_acceptance']:.3f}，burn={summaries['with_systematics']['burn']}，thin={summaries['with_systematics']['thin']}，保留{summaries['with_systematics']['flat_sample_count']}个绘图样本。

无系统项：acceptance={summaries['without_systematics']['mean_acceptance']:.3f}，burn={summaries['without_systematics']['burn']}，thin={summaries['without_systematics']['thin']}，保留{summaries['without_systematics']['flat_sample_count']}个绘图样本。

两次运行均保留原程序输出的1000步不足50倍最大自相关时间的提示，日志原样保存。本次按指定程序流程报告结果，不把它表述为通过更长链收敛认证的新测量，也不据此断言其他采样流程绝无其他分支。

## 文件

`inputs/`：作者代码、99组数据、系统协方差NPZ及原始主图快照。

`runs/with_systematics/`、`runs/without_systematics/`：两组程序、日志、原始链、实际绘图样本、摘要和原程序输出图。

`output/`：两组样本叠加三角图、局部参数对照图、参数CSV与逐项核验摘要。根目录的旧全局采样结果仅留作历史，不进入当前正文和投稿附件。
'''
    (R/'科学应用核验与结果.md').write_text(report,encoding='utf-8')
    print(json.dumps({c:summaries[c]['parameters'] for c in CASES},indent=2))

if __name__=='__main__':main()
