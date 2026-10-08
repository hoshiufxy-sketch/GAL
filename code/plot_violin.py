from pathlib import Path
import json
import math
import hashlib
import sys

OUT = Path(__file__).resolve().parent
DATA = OUT.parent / 'data'
SOURCE = DATA / 'New Data.xlsx'

def extract():
    import openpyxl
    wb = openpyxl.load_workbook(SOURCE, data_only=True, read_only=True)
    rows = list(wb['Sheet1'].values)
    records = []
    for start, group in [(0, 'Initial library'), (4, 'Round 1'), (8, 'Round 2')]:
        for rowno, row in enumerate(rows[1:], 2):
            vals = row[start:start+4]
            if all(v is None for v in vals):
                continue
            def numeric(v):
                return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
            records.append(dict(group=group, excel_row=rowno,
                                sequence=str(vals[0]).strip() if vals[0] is not None else None,
                                raw=numeric(vals[1]), od=numeric(vals[2]),
                                normalized=numeric(vals[3])))
    report = {'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'sheet': 'Sheet1', 'records': records,
              'policy': 'All finite numeric measurements retained, including missing sequence labels. No outlier filtering or replicate averaging. Normalized values as stored, not recalculated.'}
    (DATA / 'plot_data.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for g in ['Initial library', 'Round 1', 'Round 2']:
        r = [x for x in records if x['group']==g]
        print(g, {k:sum(x[k] is not None for x in r) for k in ['sequence','raw','od','normalized']})
        print('Missing:', [x for x in r if x['raw'] is None or x['sequence'] is None])

def plot():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    report = json.loads((DATA / 'plot_data.json').read_text(encoding='utf-8'))
    groups = ['Initial library', 'Round 1', 'Round 2']
    colors = ['#7894A5', '#D59769', '#B95543']
    plt.rcParams.update({'font.family':'Arial', 'font.size':11, 'axes.linewidth':1,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    stats = {}
    for key, ylabel, title in [('raw', 'Fluorescence intensity (×10$^6$ a.u.)', 'Measured fluorescence'),
                               ('normalized', 'Fluorescence / OD$_{600}$ (×10$^6$)', 'OD-normalized fluorescence')]:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        fig.subplots_adjust(left=.17, right=.97, bottom=.18, top=.84)
        arrays = [np.array([r[key] for r in report['records'] if r['group']==g and r[key] is not None])/1e6 for g in groups]
        violins = ax.violinplot(arrays, positions=[0,1,2], widths=.72,
                               showextrema=False, bw_method='scott', points=200)
        rng = np.random.default_rng(930)
        stats[key] = {}
        for i, (body, y, color) in enumerate(zip(violins['bodies'], arrays, colors)):
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.22); body.set_linewidth(1.2)
            jitter = rng.uniform(-.22, .22, len(y))
            ax.scatter(i+jitter, y, s=12 if len(y)>50 else 32, c=color,
                       alpha=.65 if len(y)>50 else .95, edgecolors='white', linewidths=.25, zorder=3)
            q1, med, q3 = np.quantile(y,[.25,.5,.75])
            ax.plot([i,i], [q1,q3], c='#26343D', lw=3.8, zorder=4, solid_capstyle='round')
            ax.scatter([i],[med], s=32, c='white', edgecolors='#26343D', lw=.9, zorder=5)
            stats[key][groups[i]] = dict(n=len(y), median=float(med*1e6), minimum=float(y.min()*1e6),maximum=float(y.max()*1e6))
        ax.set_xticks(range(3), [f'{g}\nn = {len(y)}' for g,y in zip(groups,arrays)])
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_title(title, loc='left', fontsize=13, pad=19)
        ax.set_xlim(-.6,2.6); ax.set_ylim(0,max(y.max() for y in arrays)*1.1)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',length=0,pad=10)
        ax.tick_params(axis='y',direction='out')
        ax.set_axisbelow(True); ax.grid(axis='y',color='#E9EDF0',lw=.6)
        fig.text(.17,.035,'All available measurements; median and interquartile range',fontsize=8.5,color='#59636A')
        for ext in ['png','svg','pdf']:
            fig.savefig(OUT/f'New_Data_violin_{key}.{ext}',dpi=400,facecolor='white')
        plt.close(fig)
    (DATA / 'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    extract() if '--extract' in sys.argv else plot()
