"""Preserve all displayed observations; exploratory sequence-level comparisons."""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json
import hashlib
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
DATA = OUT.parent / 'data'
data = json.loads((DATA / 'plot_data.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(data['source']).read_bytes()).hexdigest() == data['sha256'], 'Source changed: extract again first.'
records = data['records']
groups = ['Initial library', 'Round 1', 'Round 2']
colors = ['#7894A5','#D59769','#B95543']
results = []
summaries = {}
for metric in ['raw', 'normalized']:
    means = {}
    for g in groups:
        buckets = defaultdict(list)
        for r in records:
            if r['group']==g and r['sequence'] and r[metric] is not None:
                buckets[r['sequence']].append(r[metric])
        means[g] = {s:float(np.mean(v)) for s,v in buckets.items()}
    for a,b in combinations(groups,2):
        common = set(means[a]) & set(means[b])
        va = [v for s,v in means[a].items() if s not in common]
        vb = [v for s,v in means[b].items() if s not in common]
        method = 'exact' if len(set(va+vb))==len(va+vb) else 'asymptotic'
        test = mannwhitneyu(va,vb,alternative='two-sided',method=method)
        results.append(dict(metric=metric,group_a=a,group_b=b,n_a=len(va),n_b=len(vb),
                            shared_sequences_excluded=sorted(common),U=float(test.statistic),
                            p=float(test.pvalue),method=method))
    wtrows = [r for r in records if r['sequence']=='TATACC' and r[metric] is not None]
    wt = float(np.mean([r[metric] for r in wtrows]))
    maximum = max((r for r in records if r[metric] is not None),key=lambda r:r[metric])
    summaries[metric] = dict(wt_mean=wt,wt_n=len(wtrows),maximum_record=maximum,
                             maximum_fold=maximum[metric]/wt,
                             note='Maximum observation / mean WT; descriptive, not replicated effect estimate.')
# Holm adjustment over all six displayed tests (two endpoints x three pairs).
last = 0.
for rank,i in enumerate(np.argsort([r['p'] for r in results])):
    last = max(last,min(1.,(len(results)-rank)*results[i]['p']))
    results[i]['p_holm'] = last
report = dict(source=data['source'],sha256=data['sha256'],results=results,summaries=summaries,
              statistical_scope='Exploratory two-sided Mann-Whitney tests. Sequence means within each stage; sequences shared by a tested pair excluded from both groups; unidentified records excluded from tests only. Holm correction over six tests. Adaptive selection and related sequence backgrounds limit independent-sample inference; no causal GAL-vs-AL claim.',
              display='All finite numeric observations retained as in workbook. n denotes observations, not biological replicates. Stored normalized values used without imputation.')
(DATA / 'WT_statistics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.linewidth':1,
                     'svg.fonttype':'none','pdf.fonttype':42})
for metric,ylabel,title in [('raw','Fluorescence intensity (×10$^6$ a.u.)','Measured fluorescence'),
                            ('normalized','Fluorescence / OD$_{600}$ (×10$^6$)','OD-normalized fluorescence')]:
    arrays = [np.array([r[metric] for r in records if r['group']==g and r[metric] is not None])/1e6 for g in groups]
    ymax = max(a.max() for a in arrays)
    fig,ax = plt.subplots(figsize=(7.1,6.0))
    fig.subplots_adjust(left=.15,right=.97,bottom=.23,top=.90)
    vs = ax.violinplot(arrays,positions=[0,1,2],widths=.7,showextrema=False,points=200,bw_method='scott')
    rng = np.random.default_rng(930)
    for i,(y,col,body) in enumerate(zip(arrays,colors,vs['bodies'])):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(.22)
        ax.scatter(i+rng.uniform(-.20,.20,len(y)),y,s=11 if len(y)>50 else 29,
                   c=col,alpha=.68 if len(y)>50 else .95,edgecolors='white',linewidths=.3,zorder=3)
        q1,med,q3 = np.quantile(y,[.25,.5,.75])
        ax.plot([i,i],[q1,q3],color='#26343D',lw=3.7,zorder=4,solid_capstyle='round')
        ax.scatter(i,med,s=30,facecolor='white',edgecolor='#26343D',zorder=5)
    wt = summaries[metric]['wt_mean']/1e6
    ax.axhline(wt,color='#62529A',ls=(0,(5,3)),lw=1.3,zorder=2)
    ax.text(2.46,wt+.018*ymax,f'WT (TATACC): {wt:.3f} × 10⁶',ha='right',va='bottom',color='#62529A',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    mr = summaries[metric]['maximum_record']
    ax.annotate(f"Maximum: {summaries[metric]['maximum_fold']:.2f}× WT\n{mr['sequence']}",
                xy=(2,mr[metric]/1e6),xytext=(2.03,ymax*1.055),ha='center',va='bottom',fontsize=9,color='#923A2E',
                arrowprops=dict(arrowstyle='-',color='#923A2E',lw=.9))
    levels = {(0,1):1.20,(1,2):1.33,(0,2):1.46}
    for r in [r for r in results if r['metric']==metric]:
        a,b = groups.index(r['group_a']),groups.index(r['group_b'])
        yy = ymax*levels[(a,b)]; h = .025*ymax
        ax.plot([a,a,b,b],[yy,yy+h,yy+h,yy],color='#3D4248',lw=.9)
        p = r['p_holm']; label=f'{p:.2e}' if p<.001 else f'{p:.3f}'
        ax.text((a+b)/2,yy+h+.008*ymax,f'P$_{{adj}}$ = {label}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks([0,1,2],[f'{g}\nn = {len(y)}' for g,y in zip(groups,arrays)])
    ax.set_xlim(-.5,2.5); ax.set_ylim(0,ymax*1.59)
    ax.set_ylabel(ylabel); ax.set_title(title,loc='left',fontsize=13,pad=13)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=0,pad=9)
    ax.set_axisbelow(True); ax.grid(axis='y',color='#E9EDF0',lw=.6)
    fig.text(.15,.125,'Points: all measurements; white dot: median; bar: interquartile range.',fontsize=8,color='#59636A')
    fig.text(.15,.095,'Exploratory sequence-level Mann–Whitney tests; Holm correction (6 tests).',fontsize=8,color='#59636A')
    fig.text(.15,.065,'Tests exclude unidentified records and sequences shared by each compared pair.',fontsize=8,color='#59636A')
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'New_Data_violin_{metric}_WT_stats.{ext}',dpi=400,facecolor='white')
    plt.close(fig)
print(json.dumps(report,indent=2))
