"""Replot four existing Fig. 4e profiles with the paper's visual axis ranges.

This is a display-only revision. The 136 CSV values and DFT guide values are
passed to Matplotlib unchanged. Small companion axes expose off-scale values.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.tools/python_packages'), str(ROOT/'.tools/plot_packages')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_fig4_revision import load, COLORS, valid

SOURCE = ROOT/'output/strain_gaussian_image/fig4_revision'
OUT = ROOT/'output/strain_gaussian_image/paper_scale_7figures'
SPECS = [
    dict(stem='04_eps_xx_profile', field='eps_xx_percent', ref='eps_xx',
         title=r'Strain $\varepsilon_{xx}$ along the corner paths',
         label=r'Strain, $\varepsilon_{xx}$ (%)', unit='%', limits=(-2,17),
         ticks=[0,5,10,15], full=(-2,19)),
    dict(stem='05_eps_yy_profile', field='eps_yy_percent', ref='eps_yy',
         title=r'Strain $\varepsilon_{yy}$ along the corner paths',
         label=r'Strain, $\varepsilon_{yy}$ (%)', unit='%', limits=(-2,17),
         ticks=[0,5,10,15], full=(-2,19)),
    dict(stem='06_aspect_ratio_profile', field='aspect_vh', ref='aspect',
         title='Aspect ratio along the corner paths',
         label=r'Aspect ratio, $d_v/d_h$', unit='', limits=(.86,1.16),
         ticks=[.9,1,1.1], full=(.85,1.17)),
    dict(stem='07_unit_cell_area_profile', field='area_nm2', ref='area_nm2',
         title='Unit-cell area along the corner paths',
         label='Unit-cell area (nm²)', unit='nm²', limits=(.65,.77),
         ticks=[.65,.70,.75], full=(.65,.82)),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
        'axes.labelsize':16, 'xtick.labelsize':13,'ytick.labelsize':13,
        'text.color':'#222222','axes.labelcolor':'#222222',
        'axes.edgecolor':'#444444','axes.linewidth':.85,
        'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})


def arrays(rows, field):
    x = np.array([float(row['distance_cells']) for row in rows])
    y = np.array([float(row[field]) for row in rows])
    ok = np.array([valid(row) for row in rows]) & np.isfinite(y)
    return x, y, ok


def guide_name(key):
    return {'co':r'Co$_3$O$_4$', 'mn_a_sqrt2':r'Mn$_3$O$_4$ $a\sqrt{2}$',
            'mn_c':r'Mn$_3$O$_4$ $c$', 'mn_high':r'Mn$_3$O$_4$',
            'mn_low':r'Mn$_3$O$_4$', 'mn':r'Mn$_3$O$_4$'}[key]


def draw(spec, paths, guide):
    fig = plt.figure(figsize=(12.4,8.2))
    ax = fig.add_axes([.105,.295,.635,.475])
    lo,hi = spec['limits']
    all_values=[]; path_stats={}; checked_lines=0
    for key in COLORS:
        rows = paths[key]
        x,y,ok = arrays(rows,spec['field'])
        assert ok.all(), 'Unexpected missing profile values: do not bridge gaps.'
        line, = ax.plot(x,y,color=COLORS[key],lw=1.8,
            ls=':' if key=='core' else '-', marker='s' if key=='core' else 'o',
            markersize=3.4,markeredgewidth=0,label='Core' if key=='core' else f'{key} cells')
        assert np.array_equal(line.get_xdata(),x)
        assert np.array_equal(line.get_ydata(),y)
        checked_lines += 1
        lower=ok & (y<lo); upper=ok & (y>hi)
        # Triangles are boundary indicators, not substituted data points.
        for mask,boundary,mark in [(lower,lo,'v'),(upper,hi,'^')]:
            if mask.any():
                ax.scatter(x[mask],np.full(mask.sum(),boundary),s=44,marker=mark,
                    edgecolors=COLORS[key],facecolors='white',linewidths=1.1,
                    clip_on=False,zorder=9)
        path_stats[key] = dict(n=int(ok.sum()),actual_min=float(y[ok].min()),
            actual_max=float(y[ok].max()),below=int(lower.sum()),above=int(upper.sum()))
        all_values.extend(y[ok])

    outside_guides={}
    for key,num in guide.items():
        ax.axhline(num,c='#444444' if key=='co' else '#969696',ls=':',lw=1.2,zorder=0)
        if not lo<=num<=hi:
            outside_guides[key]=float(num)
            continue
        x=9.5 if key=='co' else -9.5
        # Separate Co and Mn short-axis labels even when their values are close.
        dy=-5 if key=='co' and spec['ref'].startswith('eps') else 4
        va='top' if dy<0 else 'bottom'
        if hi-num < .04*(hi-lo): dy=-5;va='top'
        ax.annotate(guide_name(key),(x,num),xytext=(0,dy),textcoords='offset points',
            ha='right' if key=='co' else 'left',va=va,fontsize=11,
            color='#444444' if key=='co' else '#777777',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.4))

    ax.axvline(0,color='#d9d9d9',lw=.8,zorder=0)
    ax.set(xlim=(-10,10),ylim=(lo,hi),xticks=[-8,-4,0,4,8],yticks=spec['ticks'],
           xlabel='Distance from corner\n(unit cells)',ylabel=spec['label'])
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=5)
    values=np.asarray(all_values)
    below=int((values<lo).sum());above=int((values>hi).sum())

    fig.text(.075,.95,spec['title'],fontsize=22,weight='bold')
    fig.text(.075,.905,'Fig. 4e display range | Gaussian projection | Final Co-core reference: 242 sites',
        fontsize=12,color='#666666')
    handles=[Line2D([],[],color=COLORS[k],lw=1.8,ls=':' if k=='core' else '-',
        marker='s' if k=='core' else None,markersize=4,label='Core' if k=='core' else f'{k} cells') for k in COLORS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.87),ncol=8,
               frameon=False,fontsize=11,columnspacing=1.4,handlelength=2)

    unit=spec['unit']; digits=4 if unit!='%' else 3
    f=lambda v: f'{v:.{digits}f}'
    fig.text(.79,.755,'Calculated values',fontsize=12,weight='bold')
    fig.text(.79,.717,f'Min  {f(values.min())} {unit}\nMax  {f(values.max())} {unit}',
             fontsize=11,linespacing=1.55,va='top')
    fig.text(.79,.637,f'Outside main axis: {below+above}/{len(values)}',fontsize=10,color='#555555')

    full_needed = bool(below+above or outside_guides)
    if full_needed:
        full = fig.add_axes([.80,.37,.17,.205])
        for key in COLORS:
            x,y,ok=arrays(paths[key],spec['field'])
            full.plot(x,y,color=COLORS[key],lw=.9,ls=':' if key=='core' else '-')
        for key,num in guide.items():
            full.axhline(num,c='#444444' if key=='co' else '#999999',lw=.6,ls=':')
        full.axhspan(lo,hi,color='#eeeeee',zorder=-1)
        full.set(xlim=(-10,10),ylim=spec['full'],xticks=[-8,0,8])
        full.set_title('Full value range',fontsize=10,pad=6)
        full.tick_params(labelsize=9,length=3)
        full.spines[['top','right']].set_visible(False)
        fig.text(.79,.329,'Gray band: main-axis range',fontsize=9,color='#666666')
    else:
        fig.text(.79,.57,'All measured points\nand reference lines\nfit within this axis.',
                 fontsize=11,color='#666666',linespacing=1.5,va='top')
    if outside_guides:
        descriptions=[]
        for key,num in outside_guides.items():
            name='Mn c' if key=='mn_c' else 'Mn area' if key=='mn' else key
            descriptions.append(f'{name}: {f(num)} {unit}')
        fig.text(.79,.285,'DFT guide above axis:\n'+'\n'.join(descriptions),fontsize=9.3,
                 color='#666666',va='top',linespacing=1.4)

    fig.text(.075,.145,f'Main axis: {lo:g} to {hi:g} {unit}. Paper axis ends reconstructed from ticks; data values are unchanged.',
        fontsize=10.5,color='#555555')
    fig.text(.075,.112,'Open triangles mark off-scale points. Dotted guides retain DFT Mn bulk / fitted Co-core values; they are not paper data.',
        fontsize=10.1,color='#555555')
    fig.text(.075,.079,'Signed path coordinate: 0 = each corner bend; 1 unit = axial valley repeat (~0.409 nm). Data exist only from -8 to +8.',
        fontsize=10.1,color='#555555')
    fig.text(.075,.046,'Source: Oh et al., Nature 577, 359–363 (2020), Fig. 4e; profiles.csv. Gaussian proxy, not multislice HAADF-STEM.',
        fontsize=9.6,color='#666666')
    for ext in ['png','svg']:
        fig.savefig(OUT/f"{spec['stem']}.{ext}",dpi=240)
    assert ax.get_xlim()==(-10,10) and ax.get_ylim()==tuple(spec['limits'])
    plt.close(fig)
    return dict(stem=spec['stem'],field=spec['field'],unit=unit,n=len(values),
        actual_min=float(values.min()),actual_max=float(values.max()),
        display_min=lo,display_max=hi,below=below,above=above,
        x_display_min=-10,x_display_max=10,x_data_min=-8,x_data_max=8,
        scale_source='Main Fig. 4e axis extent estimated from tick locations and plot borders',
        data_unchanged=True,checked_line_arrays=checked_lines,by_path=path_stats,
        reference_guide_source='Existing DFT guides, unchanged',guides_outside=outside_guides,
        full_range_companion=full_needed)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=['revised_data.npz','profiles.csv','reference_lines.json']
    before={p:sha(SOURCE/p) for p in files}
    data,paths,guides,meta=load()
    assert int(data['core_reference'].sum())==242
    assert len(paths)==8 and sum(len(rows) for rows in paths.values())==136
    style()
    report=[draw(spec,paths,guides['dft'][spec['ref']]) for spec in SPECS]
    after={p:sha(SOURCE/p) for p in files}
    assert before==after
    result=dict(source_sha256=before,source_files_unchanged=True,figures=report)
    (OUT/'profile_scale_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([{'figure':r['stem'],'actual_min':r['actual_min'],'actual_max':r['actual_max'],
                      'below':r['below'],'above':r['above']} for r in report],indent=2))


if __name__=='__main__':
    main()
