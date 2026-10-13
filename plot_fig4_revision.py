"""Fig. 4d/e layout from measured image centers and independently fitted cells.

Only the supplied revised CSV/NPZ data are plotted. Exaggerated cell outlines
are visualization glyphs; numerical strain, aspect and area are never scaled.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.tools'/'python_packages'))
sys.path.insert(0,str(ROOT/'.tools'/'plot_packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg

BASE=ROOT/'output'/'strain_gaussian_image'
OUT=BASE/'fig4_revision'
COLORS={'core':'#b332a1','+1':'#0072b2','+2':'#56b4e9','+3':'#009e73',
        '+4':'#b68b00','+5':'#d55e00','+6':'#a15395','+7':'#6e42bd'}
FIELDS=['eps_xx_percent','eps_yy_percent','aspect_vh','area_nm2']
REF_KEYS=['eps_xx','eps_yy','aspect','area_nm2']
YLABELS=[r'Strain, $\varepsilon_{xx}$ (%)',r'Strain, $\varepsilon_{yy}$ (%)',
         r'Aspect ratio, $d_v/d_h$','Unit-cell area (nm²)']


def valid(row):
    return str(row['valid']).lower() in ['true','1','yes']


def load():
    data=dict(np.load(OUT/'revised_data.npz',allow_pickle=False))
    with (OUT/'profiles.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    with (OUT/'reference_lines.json').open(encoding='utf8') as f:
        guides=json.load(f)
    with (OUT/'revision_method.json').open(encoding='utf8') as f:
        metadata=json.load(f)
    paths={}
    for row in rows:
        key=row['path_id']
        if key.lower()=='core':
            key='core'
        elif not key.startswith('+'):
            key=f'+{int(float(row["offset_cells"]))}'
        paths.setdefault(key,[]).append(row)
    for rows in paths.values():
        rows.sort(key=lambda r:float(r['distance_cells']))
    return data,paths,guides,metadata


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':12,'axes.labelsize':11,
                         'xtick.labelsize':10,'ytick.labelsize':10,
                         'axes.edgecolor':'#858585','axes.linewidth':.65,
                         'text.color':'#222222','axes.labelcolor':'#222222',
                         'figure.facecolor':'white','savefig.facecolor':'white',
                         'svg.fonttype':'none'})


def plot_paths(ax,data,paths,large):
    z=dict(np.load(BASE/'images.npz',allow_pickle=False))
    c=data['center_2d']
    x=(z['x_A']-c[0])/10; y=(z['y_A']-c[1])/10
    dx=(x[1]-x[0])/2; dy=(y[1]-y[0])/2
    im=z['image_s060']
    ax.imshow(im,origin='lower',extent=[x[0]-dx,x[-1]+dx,y[0]-dy,y[-1]+dy],
               cmap='gray',vmin=0,vmax=float(np.quantile(im,.999)),interpolation='nearest',rasterized=True)
    allxy=[]; corner=None
    H=data['B_dense_hv']
    for key,rows in paths.items():
        if key not in COLORS:
            raise ValueError(f'Unexpected path id {key}')
        xy=np.array([[float(r['x_nm']),float(r['y_nm'])] for r in rows])
        okay=np.array([valid(r) for r in rows])
        shown=xy.copy(); shown[~okay]=np.nan
        ax.plot(shown[:,0],shown[:,1],c=COLORS[key],lw=1.1 if large else .85,
                ls=':' if key=='core' else '-',zorder=5)
        allxy.extend(xy[np.isfinite(xy).all(axis=1)])
        for r,point in zip(rows,xy):
            if not valid(r):
                continue
            s=float(r['distance_cells'])
            if abs(s)<1e-10:
                ax.scatter(*point,s=15 if large else 9,color=COLORS[key],zorder=7)
                if key=='+4': corner=point
            if not np.isclose(s/2,round(s/2)):
                continue
            G=np.array([[float(r['Gxx']),float(r['Gxy'])],[float(r['Gyx']),float(r['Gyy'])]])
            E=(G+G.T)/2; W=(G-G.T)/2
            Fdisp=np.eye(2)+6*E+W
            basis=Fdisp@(4*H)*.20/10 # Angstrom to nm, display-only global scale.
            corners=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])@basis.T+point
            ax.add_patch(Polygon(corners,closed=True,fill=False,edgecolor='white',
                                 linewidth=.85 if large else .6,zorder=6))
    allxy=np.array(allxy); lo=allxy.min(axis=0)-.38; hi=allxy.max(axis=0)+.55
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xticks=[],yticks=[])
    bar_end=np.array([hi[0]-.2,lo[1]+.35]); bar_start=bar_end-np.array([2.,0.])
    ax.plot([bar_start[0],bar_end[0]],[bar_start[1],bar_end[1]],color='white',lw=2.4)
    ax.text(np.mean([bar_start[0],bar_end[0]]),bar_end[1]+.12,'2 nm',color='white',ha='center',fontsize=10 if large else 8)
    if corner is not None:
        ax.annotate('Corner cells',xy=corner,xytext=(.99,.985),textcoords='axes fraction',
                    color='#ffe345',fontsize=10 if large else 8,ha='right',va='top',
                    arrowprops={'arrowstyle':'-','color':'#ffe345','lw':.85},zorder=8)
    ax.set_xlabel('Gaussian projected image',fontsize=10)


def line_panel(ax,paths,field,ref,label,large):
    values=[]
    for key in COLORS:
        if key not in paths:
            continue
        rows=paths[key]
        s=np.array([float(r['distance_cells']) for r in rows])
        val=np.array([float(r[field]) for r in rows])
        val[~np.array([valid(r) for r in rows])]=np.nan
        ax.plot(s,val,c=COLORS[key],lw=1.25 if large else 1.0,
                ls=':' if key=='core' else '-',marker='s' if key=='core' else 'o',
                markersize=2.7 if large else 2.2,markeredgewidth=0)
        values.extend(val[np.isfinite(val)])
    for num in ref.values():
        if isinstance(num,(int,float)):
            values.append(float(num))
    lo=min(values); hi=max(values); span=max(hi-lo,.01)
    ax.set(ylim=(lo-.13*span,hi+.16*span),xlim=(-8.5,8.5),xticks=[-8,-4,0,4,8],
           xlabel='Distance from corner\n(unit cells)',ylabel=label)
    ax.axvline(0,c='#dedede',lw=.65,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    guide_labels={'co':r'Co$_3$O$_4$','mn_a_sqrt2':r'Mn$_3$O$_4$: $a\sqrt{2}$',
                  'mn_c':r'Mn$_3$O$_4$: $c$', 'mn_high':r'Mn$_3$O$_4$',
                  'mn_low':r'Mn$_3$O$_4$', 'mn':r'Mn$_3$O$_4$'}
    for key,num in ref.items():
        if key not in guide_labels:
            continue
        ax.axhline(num,color='#7e7e7e' if key!='co' else '#353535',ls=':',lw=.85,zorder=0)
        # Opposite alignment separates nearly coincident Co and Mn-short guides.
        right=key=='co'
        ax.annotate(guide_labels[key],xy=(8.15 if right else -8.15,float(num)),
                    xytext=(0,3),textcoords='offset points',ha='right' if right else 'left',
                    va='bottom',fontsize=8.2 if large else 6.8,color='#454545' if right else '#757575',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':.3})


def draw(data,paths,allguides,kind,readable=False):
    guide=allguides[kind]
    if readable:
        fig=plt.figure(figsize=(16,9.5))
        gs=fig.add_gridspec(2,3,left=.045,right=.985,top=.78,bottom=.265,
                            width_ratios=[1.07,1,1],wspace=.36,hspace=.40)
        path_ax=fig.add_subplot(gs[:,0])
        axes=[fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2])]
    else:
        fig=plt.figure(figsize=(19.5,6.1))
        gs=fig.add_gridspec(1,5,left=.025,right=.99,top=.72,bottom=.35,
                            width_ratios=[1.1,1,1,1,1],wspace=.43)
        path_ax=fig.add_subplot(gs[0,0]); axes=[fig.add_subplot(gs[0,i]) for i in range(1,5)]
    plot_paths(path_ax,data,paths,readable)
    path_ax.set_title('d  Measured paths and cell glyphs',fontsize=11 if readable else 9,loc='left',pad=9)
    for i,(ax,field,refkey,label) in enumerate(zip(axes,FIELDS,REF_KEYS,YLABELS)):
        line_panel(ax,paths,field,guide[refkey],label,readable)
        if i==0:
            ax.text(-.22,1.15,'e',transform=ax.transAxes,weight='bold',fontsize=13)
    normal_limits=(min(axes[0].get_ylim()[0],axes[1].get_ylim()[0]),
                   max(axes[0].get_ylim()[1],axes[1].get_ylim()[1]))
    axes[0].set_ylim(normal_limits); axes[1].set_ylim(normal_limits)
    left=.045 if readable else .025
    fig.text(left,.955,'Fig. 4d/e reconstruction — Gaussian-projected model',fontsize=19,weight='bold')
    source='DFT-relaxed Mn₃O₄ bulk' if kind=='dft' else 'Literature Mn₃O₄ bulk'
    fig.text(left,.899,f'Gaussian image fitting · final Co-core reference · {source} comparison guides',fontsize=12,color='#555555')
    handles=[]
    for key in COLORS:
        if key in paths:
            name='Core' if key=='core' else f'{key} cells'
            handles.append(Line2D([],[],c=COLORS[key],ls=':' if key=='core' else '-',
                                  marker='s' if key=='core' else None,markersize=3,lw=1.7,label=name))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.86),ncol=8,frameon=False,fontsize=10)
    H=data['B_dense_hv']; cell_nm=float(np.mean(np.linalg.norm(2*H,axis=0)))/10
    if readable:
        ys=[.128,.094,.060,.026]
    else:
        ys=[.173,.127,.081,.034]
    fig.text(left,ys[0],f'1 unit here = axial valley repeat (~{cell_nm:.3f} nm), not diagonal primitive spacing. Zero is each bend; anchor is a chosen inner core-corner site, not the interface corner.',fontsize=9.8,color='#555555')
    fig.text(left,ys[1],'White cell glyphs: strain ×6, rotation ×1, overall symbol scale 0.20; display only. Curves retain unscaled measured values. Missing fits/support break curves.',fontsize=9.8,color='#555555')
    fig.text(left,ys[2],'Aspect ratio is raw dv/dh; area is conventional-cell-equivalent projected area. All dotted Mn guides are converted to the fitted Co reference where required.',fontsize=9.8,color='#555555')
    fig.text(left,ys[3],'Reconstructed analysis and drawing choices: original author MATLAB code unavailable. Reference: Oh et al., Nature 577 (2020), Fig. 4 and SI pp. 3–4. Not multislice HAADF.',fontsize=9.2,color='#555555')
    stem=f'fig4de_{kind}'+('_readable' if readable else '')
    for suffix in ['png','svg']:
        fig.savefig(OUT/f'{stem}.{suffix}',dpi=230)
    plt.close(fig)


def comparison():
    source=ROOT/'tmp'/'ppt_strain'/'assets'/'source'/'paper_fig4de_paths_profiles.png'
    if not source.exists():
        return
    paper=mpimg.imread(source); ours=mpimg.imread(OUT/'fig4de_dft.png')
    fig=plt.figure(figsize=(19.5,11))
    a=fig.add_axes([.025,.58,.95,.34]); a.imshow(paper); a.axis('off')
    b=fig.add_axes([.025,.035,.95,.49]); b.imshow(ours); b.axis('off')
    fig.text(.025,.947,'Paper HAADF-STEM',fontsize=19,weight='bold')
    fig.text(.025,.554,'This model: Gaussian projection',fontsize=19,weight='bold')
    fig.text(.985,.948,'Oh et al., Nature 577, 359–363 (2020), Fig. 4d/e',fontsize=11,ha='right',color='#555555')
    fig.savefig(OUT/'paper_vs_gaussian_fig4de.png',dpi=180)
    plt.close(fig)


def main():
    style(); data,paths,guides,meta=load()
    for kind in ['dft','literature']:
        if kind in guides:
            draw(data,paths,guides,kind,False)
            draw(data,paths,guides,kind,True)
    comparison()
    print(json.dumps({'figures':sorted(p.name for p in OUT.glob('*.png')),
                      'path_ids':list(paths),'source_csv_rows':sum(map(len,paths.values()))},indent=2))


if __name__=='__main__':
    main()
