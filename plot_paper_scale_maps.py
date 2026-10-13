"""Display saved Gaussian-image strain with the published Fig. 4c color scale.

No strain, reference, positions or masks are recomputed. The palette is sampled
from the supplied published color-bar raster, not the authors' original LUT.
"""
from __future__ import annotations
import hashlib
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from PIL import Image

from plot_gaussian_image_strain import load_data, rejected_markers, core_outline, spatial_axes

SOURCE=ROOT/'output'/'strain_gaussian_image'/'fig4_revision'/'revised_data.npz'
PALETTE_SOURCE=ROOT/'tmp'/'ppt_strain'/'assets'/'source'/'paper_fig4c_strain_rotation.png'
OUT=ROOT/'output'/'strain_gaussian_image'/'paper_scale_7figures'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(a):
    h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def palette():
    image=np.asarray(Image.open(PALETTE_SOURCE).convert('RGB'))
    if image.shape[:2]!=(666,1875):
        raise ValueError(f'Unexpected source raster geometry: {image.shape}')
    # Excludes the white border and all ticks/labels: inclusive pixel rectangle.
    band=image[87:104,53:246,:]
    rgb=np.median(band,axis=0)/255.
    cmap=LinearSegmentedColormap.from_list('published_Fig4c_sampled',rgb,N=1024)
    cmap.set_under(rgb[0]);cmap.set_over(rgb[-1])
    return cmap,{'source_path':str(PALETTE_SOURCE),'source_sha256':sha256(PALETTE_SOURCE),
                 'sampling_rectangle_inclusive_xy':[53,87,245,103],
                 'method':'Per-column median RGB over the interior color bar; linear interpolation of 193 samples to a 1024-entry Matplotlib LUT.',
                 'author_original_lut':False,
                 'rgb_8bit':np.rint(rgb*255).astype(int).tolist()}


def draw(d,raw,cmap,field,multiplier,vmin,vmax,ticks,stem,title,unit):
    original_valid=raw['valid'].astype(bool)
    values=raw[field]*multiplier
    mask=d['valid'] & np.isfinite(values)
    # Only unit conversion is permitted. No changed eligibility or filtering.
    assert np.array_equal(mask,original_valid)
    v=values[mask]
    low=int(np.count_nonzero(v<vmin)); high=int(np.count_nonzero(v>vmax))
    fig=plt.figure(figsize=(10,9.8))
    ax=fig.add_axes([.095,.258,.74,.635])
    rejected_markers(ax,d,small=False)
    scatter=ax.scatter(d['xy'][mask,0],d['xy'][mask,1],c=v,
                       cmap=cmap,norm=Normalize(vmin=vmin,vmax=vmax,clip=False),
                       s=20,marker='o',linewidths=0,rasterized=True)
    core_outline(ax,d); spatial_axes(ax,d)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Image x // [010] (nm)',fontsize=12)
    ax.set_ylabel('Image y // [001] (nm)',fontsize=12)
    # Matplotlib must receive the exact converted saved values, not clipped data.
    assert np.array_equal(np.asarray(scatter.get_array()),v)
    assert np.array_equal(np.asarray(scatter.get_offsets()),d['xy'][mask])
    cax=fig.add_axes([.85,.285,.026,.58])
    cb=fig.colorbar(scatter,cax=cax,ticks=ticks,
                    extend='both' if low+high else 'neither',extendfrac=.035)
    cb.set_label(unit,fontsize=12,labelpad=9)
    cb.ax.tick_params(labelsize=11)
    fig.text(.065,.959,title,fontsize=21,weight='bold')
    fig.text(.065,.918,'Gaussian projected image | Final Co-core reference | [100] projection',fontsize=11,color='#555555')
    ncore=int(d['core_reference'].sum())
    handles=[Line2D([],[],marker='x',color='#aaaaaa',ls='none',markersize=5,label='Rejected image fit'),
             Line2D([],[],marker='o',mfc='none',mec='#aaaaaa',ls='none',markersize=5,label='Unsupported derivative'),
             Line2D([],[],color='black',ls=(0,(4,3)),lw=.9,label=f'Core reference region ({ncore} sites)')]
    legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.49,.172),ncol=3,
                      frameon=False,fontsize=9.1,handlelength=2.,columnspacing=1.3)
    suffix='%' if unit=='Strain (%)' else 'deg'
    fig.text(.065,.153,f'Display: {vmin:g} to {vmax:g} {suffix} | Actual valid range: {v.min():.3f} to {v.max():.3f} {suffix}',fontsize=11)
    fig.text(.065,.123,f'Below display range: {low:,} | Above display range: {high:,} | Colored sites: {len(v):,}/{len(mask):,}',fontsize=11)
    fig.text(.065,.091,'Endpoint colors represent out-of-range values; the saved values and point positions are unchanged.',fontsize=9.8,color='#555555')
    fig.text(.065,.061,'Palette sampled from published Fig. 4c, not the original author LUT. No spatial interpolation.',fontsize=9.8,color='#555555')
    fig.text(.065,.031,'Gaussian projection proxy, not multislice HAADF-STEM. Source: Oh et al., Nature 577 (2020), Fig. 4c.',fontsize=9.5,color='#555555')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    assert not ax.xaxis.label.get_window_extent(renderer).overlaps(legend.get_window_extent(renderer))
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=230)
    plt.close(fig)
    return {'field':field,'output_stem':stem,'source_unit_to_display_multiplier':multiplier,
            'display_unit':suffix,'normalization':'matplotlib.colors.Normalize, linear',
            'display_vmin':vmin,'display_vmax':vmax,'ticks':ticks,
            'actual_valid_min':float(v.min()),'actual_valid_max':float(v.max()),
            'below_count':low,'above_count':high,'valid_count':len(v),'total_count':len(mask),
            'core_reference_count':ncore,'colorbar_extension':'both' if low+high else 'neither',
            'xlabel_legend_overlap_check_passed':True,
            'scatter_values_exactly_equal_saved_values_after_unit_conversion':True,
            'scatter_positions_exactly_equal_saved_positions_after_centering_and_nm_conversion':True,
            'valid_mask_exactly_equal_saved_mask':True}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.edgecolor':'#888888','axes.linewidth':.65,
                         'axes.labelcolor':'#222222','text.color':'#222222',
                         'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    before=sha256(SOURCE)
    raw=dict(np.load(SOURCE,allow_pickle=False))
    hashes_before={key:array_digest(a) for key,a in raw.items()}
    d=load_data(SOURCE)
    cmap,palette_meta=palette()
    configs=[('eps_xx',100.,-4.,16.,[-4,0,6,10,16],'01_eps_xx_map',r'Strain, $\varepsilon_{xx}$ — paper color scale','Strain (%)'),
             ('eps_yy',100.,-4.,16.,[-4,0,6,10,16],'02_eps_yy_map',r'Strain, $\varepsilon_{yy}$ — paper color scale','Strain (%)'),
             ('rotation_deg',1.,-2.3,2.3,[-2.3,0,2.3],'03_rotation_map',r'Rotation, $\theta$ — paper color scale','Rotation (deg)')]
    outputs=[draw(d,raw,cmap,*args) for args in configs]
    after=sha256(SOURCE)
    arrays_after=dict(np.load(SOURCE,allow_pickle=False))
    hashes_after={key:array_digest(a) for key,a in arrays_after.items()}
    assert before==after
    assert hashes_before==hashes_after
    report={'source_npz':str(SOURCE),'source_sha256_before':before,'source_sha256_after':after,
            'source_file_unchanged':True,'all_saved_arrays_unchanged':True,
            'saved_array_sha256':hashes_before,'palette':palette_meta,'figures':outputs,
            'scope':'Display range and colormap only. No recomputation, image-physics changes, spatial interpolation, ROI mask change, or geometric rescaling.'}
    (OUT/'map_scale_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'source_file_unchanged':True,'all_arrays_unchanged':True,'figures':outputs},indent=2))


if __name__=='__main__':
    main()
