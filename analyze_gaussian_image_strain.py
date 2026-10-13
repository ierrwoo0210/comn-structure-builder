"""Image-valley strain from a Gaussian projected signal, NOT multislice HAADF.

Final coordinate proxies provide search seeds and lattice identifiers only.
All fitted positions are measured independently from the generated intensity.
"""
from pathlib import Path
import sys, json, csv, argparse, time, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.tools/python_packages')]
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from analyze_paper_strain import reference_fit, gradient_fit, analytic_checks, A0

OUT = ROOT / 'output/strain_gaussian_image'
CASES = [('s045', .45, 'image_s045'), ('s060', .60, 'image_s060'),
         ('s075', .75, 'image_s075'), ('s060_p200', .60, 'image_s060_p200')]
LOWPASS = .25
PATCH = 1.20
BOUND = .80

def model(p, uv, jac=False):
    b, a, cx, cy, sx, sy, rho = p
    u, v = uv[:, 0]-cx, uv[:, 1]-cy
    qx, qy = u/sx, v/sy
    den = 1-rho*rho
    num = qx*qx+qy*qy-2*rho*qx*qy
    g = np.exp(-.5*num/den)
    f = b-a*g
    if not jac:
        return f
    out = np.empty((len(u), 7))
    out[:, 0] = 1
    out[:, 1] = -g
    out[:, 2] = -a*g*(u/sx**2-rho*v/(sx*sy))/den
    out[:, 3] = -a*g*(v/sy**2-rho*u/(sx*sy))/den
    out[:, 4] = -a*g*(u*u/sx**3-rho*u*v/(sx*sx*sy))/den
    out[:, 5] = -a*g*(v*v/sy**3-rho*u*v/(sx*sy*sy))/den
    out[:, 6] = -a*g*(qx*qy/den-rho*num/den**2)
    return out

LO = np.array([-1., .01, -BOUND, -BOUND, .15, .15, -.85])
HI = np.array([6., 8., BOUND, BOUND, 2., 2., .85])

def fit_one(image, x, y, seed):
    ix = np.where(abs(x-seed[0]) <= PATCH)[0]
    iy = np.where(abs(y-seed[1]) <= PATCH)[0]
    xx, yy = np.meshgrid(x[ix]-seed[0], y[iy]-seed[1])
    uv = np.c_[xx.ravel(), yy.ravel()]
    val = image[np.ix_(iy, ix)].ravel().astype(float)
    base, top = np.percentile(val, [2, 98])
    span = top-base
    if span < 1e-10:
        return np.full(7, np.nan), np.nan, span, False, 0
    norm = (val-base)/span
    search = np.linalg.norm(uv, axis=1) < .55
    minid = np.where(search)[0][np.argmin(norm[search])]
    start = np.array([1.2, 1.2, *uv[minid], .65, .65, 0.])
    fit = least_squares(lambda p: model(p, uv)-norm, start,
        jac=lambda p: model(p, uv, True), bounds=(LO, HI), max_nfev=70,
        ftol=1e-7, xtol=1e-7, gtol=1e-7)
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    p = fit.x
    boundary = bool(np.any((p-LO) < .01*(HI-LO)) or np.any((HI-p) < .01*(HI-LO)))
    valid = bool(fit.success and rms <= .16 and not boundary and p[1] > .15)
    return p, rms, span, valid, int(fit.nfev)

def numerical_checks():
    grid = np.linspace(-PATCH, PATCH, 25)
    xx, yy = np.meshgrid(grid, grid)
    uv = np.c_[xx.ravel(), yy.ravel()]
    p = np.array([1.2, 1., .23, -.17, .55, .72, .2])
    eps = 1e-6
    finite = np.column_stack([(model(p+np.eye(7)[j]*eps, uv)-model(p-np.eye(7)[j]*eps, uv))/(2*eps) for j in range(7)])
    jac_error = float(np.max(abs(finite-model(p, uv, True))))
    assert jac_error < 1e-7
    fit = least_squares(lambda q: model(q, uv)-model(p, uv), [1.1,.9,0,0,.65,.65,0],
                        jac=lambda q:model(q,uv,True), bounds=(LO,HI),gtol=1e-12,ftol=1e-12,xtol=1e-12)
    err = float(np.max(abs(fit.x-p)))
    assert err < 1e-8
    return dict(analytic_jacobian_max_abs_error=jac_error, synthetic_gaussian_parameter_max_abs_error=err,
                affine_displacement_gradient_checks=analytic_checks())

def fit_case(images, old, name, sigma, key, limit=0):
    target = OUT/f'valley_fits_{name}.npz'
    signature = hashlib.sha256(json.dumps(dict(
        image_metadata_sha256=hashlib.sha256((OUT/'image_generation_metadata.json').read_bytes()).hexdigest(),
        image_key=key,image_shape=images[key].shape,sigma=sigma,lowpass=LOWPASS,patch=PATCH,
        lower=LO.tolist(),upper=HI.tolist(),nrmse_max=.16,amplitude_min=.15,
        bound_fraction=.01,duplicate_distance_A=1.,max_nfev=70,
        seed_sha256=hashlib.sha256(old['positions'].tobytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
    if target.exists() and not limit:
        cached = dict(np.load(target))
        if str(cached.get('cache_signature','')) == signature:
            return cached
    x, y = images['x_A'], images['y_A']
    pixel = float(x[1]-x[0])
    image = gaussian_filter(images[key], LOWPASS/pixel, mode='constant', truncate=5.)
    seeds = old['positions']
    ids = np.arange(len(seeds)) if not limit else np.unique(np.linspace(0,len(seeds)-1,limit).astype(int))
    pars = np.full((len(seeds),7),np.nan)
    rms, contrast = np.full(len(seeds),np.nan), np.full(len(seeds),np.nan)
    accepted = np.zeros(len(seeds),bool)
    nfev = np.zeros(len(seeds),int)
    t0=time.time()
    for n,i in enumerate(ids):
        pars[i],rms[i],contrast[i],accepted[i],nfev[i] = fit_one(image,x,y,seeds[i])
        if (n+1)%1000==0:
            print(f'{name}: fitted {n+1}/{len(ids)}, accepted {accepted.sum()}, seconds {time.time()-t0:.1f}',flush=True)
    pos = seeds+pars[:,2:4]
    # Reject both ambiguous assignments rather than silently merge sites.
    good = np.where(accepted)[0]
    pairs=cKDTree(pos[good]).query_pairs(1.0,output_type='ndarray') if len(good)>1 else np.empty((0,2),int)
    if len(pairs):
        accepted[good[np.unique(pairs)]]=False
    result=dict(positions=pos,parameters=pars,fit_nrmse=rms,fit_contrast=contrast,fit_valid=accepted,
        fit_shift_A=np.linalg.norm(pars[:,2:4],axis=1),fit_nfev=nfev,duplicate_pairs=len(pairs),
        sigma_image_A=sigma,sigma_lowpass_A=LOWPASS,effective_sigma_A=np.hypot(sigma,LOWPASS),
        cache_signature=signature)
    if not limit: np.savez_compressed(target,**result)
    print(f'{name}: accepted {accepted.sum()}/{len(ids)}, median NRMSE {np.nanmedian(rms):.4f}',flush=True)
    return result

def supported_gradient(ref, pos, B, fit_valid, query=None):
    query=ref if query is None else query
    rr,pp=ref[fit_valid],pos[fit_valid]
    g=gradient_fit(rr,pp,B,1.5,query=query)
    spacing=np.mean(np.linalg.norm(B,axis=0))
    sigma=1.5*spacing
    tree=cKDTree(rr)
    nearest=tree.query(query)[0]
    neighborhoods=tree.query_ball_point(query,3.5*sigma)
    bias=np.full(len(query),np.inf)
    cond=np.full(len(query),np.inf)
    close_count=np.zeros(len(query),int)
    for i,ids in enumerate(neighborhoods):
        dv=rr[ids]-query[i]
        w=np.exp(-np.sum(dv*dv,axis=1)/(2*sigma*sigma))
        if len(ids)<3: continue
        avg=np.sum(w[:,None]*dv,axis=0)/w.sum()
        bias[i]=np.linalg.norm(avg)/sigma
        cov=((dv-avg).T*w)@(dv-avg)/w.sum()
        eig=np.linalg.eigvalsh(cov)
        cond[i]=eig[-1]/max(eig[0],1e-20)
        close_count[i]=np.sum(np.linalg.norm(dv,axis=1)<1.5*spacing)
    support=(nearest<=.75*spacing)&(g['effective_neighbors']>=12)&(bias<=.75)&(cond<=6)&(close_count>=6)
    g.update(support=support,nearest_fit_A=nearest,support_centroid_bias_sigma=bias,support_condition=cond,nearby_fit_count=close_count)
    g['valid'] &= support
    return g

def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def compute_result(old, fit, core):
    pos,labels=fit['positions'],old['labels']
    ref,B,H,coeff=reference_fit(labels,pos,core)
    g=supported_gradient(ref,pos,B,fit['fit_valid'])
    # derivative can be evaluated at missing sites but must not be reported as measured.
    g['valid'] &= fit['fit_valid']
    data={k:old[k] for k in ['labels','region','topology_positions','core_fraction','core_interface_clearance_A','beam_overlap_fraction']}
    data.update(fit)
    data.update({k:v for k,v in g.items() if k not in ['displacement']})
    data.update(reference=ref,seed_positions=old['positions'],B_primitive=B,B_dense_hv=H,
        center_2d=np.r_[1,4*(.318-.125),4*(.318-.125)]@coeff,
        core_reference=core,displacement=pos-ref,beam_axis=0)
    J=np.eye(2)+g['G']
    data['area_nm2']=8*np.linalg.det(J@B)/100
    data['aspect_vh']=np.linalg.norm(J@H[:,1],axis=1)/np.linalg.norm(J@H[:,0],axis=1)/(np.linalg.norm(H[:,1])/np.linalg.norm(H[:,0]))
    return data

def profiles(data, offsets=range(1,8)):
    B,H=data['B_primitive'],data['B_dense_hv']
    spacing=np.mean(np.linalg.norm(B,axis=0))
    hx,vy=H[:,0]/np.linalg.norm(H[:,0]),H[:,1]/np.linalg.norm(H[:,1])
    corner=data['center_2d']+H@np.array([55/(A0/4),55/(A0/4)])
    rows=[]
    for offset in offsets:
        s=np.linspace(-8,8,129)
        q=np.tile(corner+offset*spacing*(hx+vy),(len(s),1))
        q[s<=0]+=s[s<=0,None]*spacing*hx
        q[s>0]-=s[s>0,None]*spacing*vy
        g=supported_gradient(data['reference'],data['positions'],B,data['fit_valid'],q)
        observed=q+g['smooth_displacement']
        J=np.eye(2)+g['G']
        ar=np.linalg.norm(J@H[:,1],axis=1)/np.linalg.norm(J@H[:,0],axis=1)/(np.linalg.norm(H[:,1])/np.linalg.norm(H[:,0]))
        area=8*np.linalg.det(J@B)/100
        for i in range(len(s)):
            rows.append(dict(offset_cells=offset,distance_cells=float(s[i]),
                reference_x_A=float(q[i,0]),reference_y_A=float(q[i,1]),
                observed_x_A=float(observed[i,0]) if g['valid'][i] else np.nan,
                observed_y_A=float(observed[i,1]) if g['valid'][i] else np.nan,
                eps_xx_percent=float(g['eps_xx'][i]*100) if g['valid'][i] else np.nan,
                eps_yy_percent=float(g['eps_yy'][i]*100) if g['valid'][i] else np.nan,
                eps_xy_percent=float(g['eps_xy'][i]*100) if g['valid'][i] else np.nan,
                rotation_deg=float(g['rotation_deg'][i]) if g['valid'][i] else np.nan,
                aspect_vh=float(ar[i]) if g['valid'][i] else np.nan,
                area_nm2=float(area[i]) if g['valid'][i] else np.nan,valid=bool(g['valid'][i]),
                effective_neighbors=float(g['effective_neighbors'][i]),nearest_fit_A=float(g['nearest_fit_A'][i])))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=int,default=0)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    checks=numerical_checks()
    images=np.load(OUT/'images.npz')
    old=dict(np.load(ROOT/'output/strain_paper_method/paper_projection.npz'))
    fits={name:fit_case(images,old,name,sigma,key,args.pilot) for name,sigma,key in CASES if key in images.files}
    if args.pilot:return
    common_core=old['core_reference'].copy()
    for fit in fits.values():common_core &= fit['fit_valid']
    assert common_core.sum()>=100,'Too few shared image-fitted deep-core sites'
    # The clearance is a structural prior; use the largest measured spacing across cases.
    for _ in range(3):
        max_spacing=max(np.mean(np.linalg.norm(reference_fit(old['labels'],f['positions'],common_core)[1],axis=0)) for f in fits.values())
        common_core &= old['core_interface_clearance_A']>=4*max_spacing
    result={name:compute_result(old,fit,common_core) for name,fit in fits.items()}
    for name,d in result.items():np.savez_compressed(OUT/f'gaussian_strain_{name}.npz',**d)
    primary=result['s060']
    np.savez_compressed(OUT/'gaussian_strain.npz',**primary)
    rows=[]
    for i in range(len(primary['positions'])):
        rows.append(dict(cell_id=i,x_A=float(primary['positions'][i,0]),y_A=float(primary['positions'][i,1]),
            fit_valid=bool(primary['fit_valid'][i]),strain_valid=bool(primary['valid'][i]),
            region=int(primary['region'][i]),core_reference=bool(common_core[i]),
            eps_xx_percent=float(primary['eps_xx'][i]*100) if primary['valid'][i] else np.nan,
            eps_yy_percent=float(primary['eps_yy'][i]*100) if primary['valid'][i] else np.nan,
            eps_xy_percent=float(primary['eps_xy'][i]*100) if primary['valid'][i] else np.nan,
            rotation_deg=float(primary['rotation_deg'][i]) if primary['valid'][i] else np.nan,
            displacement_x_A=float(primary['displacement'][i,0]) if primary['fit_valid'][i] else np.nan,
            displacement_y_A=float(primary['displacement'][i,1]) if primary['fit_valid'][i] else np.nan,
            aspect_vh=float(primary['aspect_vh'][i]) if primary['valid'][i] else np.nan,
            area_nm2=float(primary['area_nm2'][i]) if primary['valid'][i] else np.nan,
            fit_nrmse=float(primary['fit_nrmse'][i]),fit_shift_A=float(primary['fit_shift_A'][i])))
    save_csv(OUT/'strain_points.csv',rows)
    save_csv(OUT/'corner_profiles.csv',profiles(primary))
    sensitivity=[]
    for name,d in result.items():
        for region_name,mask in [('all',np.ones(len(common_core),bool)),('core',old['region']==1),('CL',old['region']==2),('island',old['region']==3),('beam_overlap_shell',(old['beam_overlap_fraction']>0)&(old['region']!=1))]:
            m=mask&d['valid']&primary['valid']
            for key in ['eps_xx','eps_yy','eps_xy']:
                delta=100*abs(d[key][m]-primary[key][m])
                sensitivity.append(dict(case=name,region=region_name,component=key,
                    valid_n=int((mask&d['valid']).sum()),common_valid_n=int(m.sum()),
                    median_abs_diff_percentage_points=float(np.median(delta)) if len(delta) else None,
                    p90_abs_diff_percentage_points=float(np.quantile(delta,.9)) if len(delta) else None))
    save_csv(OUT/'sigma_sensitivity.csv',sensitivity)
    summary=dict(method='Gaussian projected intensity -> negative 2D Gaussian valley fitting -> final deep-core reference -> infinitesimal apparent projected strain',
        not_multislice_or_physical_HAADF=True,beam='[100]',horizontal='[010]',vertical='[001]',
        atom_signal='All individual final atoms; (Z/27)^1.7 heuristic amplitude, p2.0 sensitivity; no depth weighting and no pre-averaged columns.',
        signal_sigmas_A=[.45,.60,.75],primary_sigma_A=.60,additional_lowpass_sigma_A=LOWPASS,
        effective_primary_sigma_A=float(np.hypot(.60,LOWPASS)),
        width_selection='Pilot80 uniformly indexed seeds: accepted9 at sigma.30,45 at.45,77 at.60; narrow-image shell valleys hit Gaussian width bounds. Choose.60 as primary; validate.45/.75 sensitivity. Pilot-based fit identifiability, not instrument calibration.',
        seeds='4782 final-coordinate hole-center proxies used ONLY as search seeds, initial topology as identifiers. Not blind peak detection.',
        valley_fit='Constant background minus positive elliptical Gaussian; center limited +/-0.8 A about final proxy seed; square patch halfwidth1.2 A; sigma bounds .15..2 A and rho +/-0.85.',
        acceptance='NRMSE<=0.16 (patch p98-p2 normalization), optimizer success, parameters >1% away from bounds, amplitude>0.15, no accepted center pair closer than1A. Thresholds are analyst-selected diagnostics, not experimental error bars.',
        derivative='Gaussian weighted local linear displacement regression, sigma1.5 primitive cells; infinitesimal tensor and rotation; invalid fits excluded from all neighbors.',
        derivative_support='effective neighbors>=12; nearest fit<=0.75spacing; >=6 accepted centers within1.5spacing; weighted spatial covariance condition<=6; weighted neighborhood centroid offset<=0.75sigma; positive deformation Jacobian. Missing measured sites not displayed.',
        core_reference='Shared accepted core labels across all image cases; structural lateral-interface clearance>=4 measured primitive spacings; final image positions set each case reference metric. Front/back shell signals remain included.',
        old_overlap_mask_not_used=True,
        corner_offsets='Upper-right CORE corner, +1..+7 measured primitive spacings along both image axes. Zero distance is each path bend; CL starts at core boundary; offsets are chosen for this model, not paper scan paths.',
        profile_correlation='Adjacent paths overlap within derivative smoothing kernel; not independent observations.',
        source='Oh et al., Nature577,359-363(2020),doi10.1038/s41586-019-1899-3; SI strain measurements printedpp3-4. Gaussian image creation is additional approximation.',
        limits=['No electron multiple scattering, channeling, detector response, specimen tilt, phonon dynamics or experimental dose/noise.',
                '2D apparent strain relative to Co core includes intrinsic Co/Mn mismatch, not bulk-Mn residual elastic strain.',
                'Acceptance shows numerical fit quality, not validation of HAADF image physics. Width/amplitude sensitivity and incomplete coverage must accompany interpretation.'],
        numerical_checks=checks,shared_core_reference_n=int(common_core.sum()),cases={})
    for name,d in result.items():
        valid=d['valid'];m=valid&old['roi_lateral']
        summary['cases'][name]=dict(fit_valid_n=int(d['fit_valid'].sum()),strain_valid_n=int(valid.sum()),
            seed_n=len(valid),core_basis_A=d['B_primitive'].tolist(),primitive_spacing_A=float(np.mean(np.linalg.norm(d['B_primitive'],axis=0))),
            fit_nrmse_median=float(np.median(d['fit_nrmse'][d['fit_valid']])),
            fit_shift_A_median=float(np.median(d['fit_shift_A'][d['fit_valid']])),
            old_coordinate_comparison={k:dict(common_lateral_n=int(m.sum()),median_abs_difference_pp=float(np.median(abs(d[k][m]-old[k][m]))*100)) for k in ['eps_xx','eps_yy']})
    (OUT/'method_and_validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary['cases'],indent=2),flush=True)

if __name__=='__main__':main()
