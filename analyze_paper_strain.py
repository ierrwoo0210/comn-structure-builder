"""Paper-reference structural counterpart of Fig.3/4 from full-depth XYZ projection.

No DFT lattice and no initial-final strain. Initial coordinates supply integer
spinel topology only. Final deep projected core defines the metric reference.
Coordinate-derived empty-hole centers replace HAADF negative-Gaussian valleys.
"""
from pathlib import Path
import sys
import json
import csv
import argparse
import itertools
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python_packages'))
import numpy as np
from scipy.spatial import cKDTree

OUT = ROOT / 'output/strain_paper_method'
A0 = 8.084
INITIAL = Path('C:/Users/riphy/OneDrive/바탕 화면/Co3O4_Mn3O4_core11nm_CL3nm_island2nm.xyz')
FINAL = Path('C:/Users/riphy/OneDrive/바탕 화면/final_11nm.xyz')


def read_xyz(path):
    with path.open(encoding='utf-8') as f:
        count = int(f.readline())
        header = f.readline().strip()
        a = np.loadtxt(f, dtype=[('species', 'U2'), ('x', 'f8'), ('y', 'f8'),
                                ('z', 'f8'), ('region', 'i4')], usecols=(0, 1, 2, 3, 4))
    assert len(a) == count, 'Only a single complete frame is accepted'
    assert 'pbc="F F F"' in header, 'This implementation is nonperiodic'
    pos = np.column_stack([a[k] for k in ['x', 'y', 'z']])
    assert np.isfinite(pos).all()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return pos, a['species'], a['region'], dict(path=str(path), sha256=sha, header=header)


def aggregate_columns(X, Y, species, region, selected, image_axes, sublattice):
    """Group original crystallographic columns, then project all final atoms in depth."""
    xyz0, xyz = X[selected], Y[selected]
    frac_unwrapped = (xyz0 - 117) / A0 + .318
    shift = .125 if sublattice == 'Td' else 0
    labels = np.rint((frac_unwrapped[:, image_axes] - shift) * 4).astype(int)
    assert np.max(np.abs((frac_unwrapped[:, image_axes] - shift) * 4 - labels)) < 1e-6
    unique, inv = np.unique(labels, axis=0, return_inverse=True)
    count = np.bincount(inv)
    pos = np.column_stack([np.bincount(inv, weights=xyz[:, j]) / count for j in image_axes])
    residual = xyz[:, image_axes] - pos[inv]
    spread = np.sqrt(np.bincount(inv, weights=np.sum(residual**2, axis=1)) / count)
    core_count = np.bincount(inv, weights=region[selected] == 1)
    core_pos = np.column_stack([np.divide(np.bincount(inv, weights=xyz[:, j] * (region[selected] == 1)), core_count,
                                         out=np.full(len(count), np.nan), where=core_count > 0) for j in image_axes])
    beam = ({0, 1, 2} - set(image_axes)).pop()
    q = np.abs(xyz0 - 117)
    front = (region[selected] != 1) & (q[:, beam] > np.max(q[:, image_axes], axis=1) + 1e-7)
    front_frac = np.bincount(inv, weights=front) / count
    return dict(labels=unique, pos=pos, core_pos=core_pos, count=count, spread=spread,
                core_fraction=core_count / count, front_fraction=front_frac)


def projected_centers(X, Y, species, region, beam=0):
    image_axes = [i for i in range(3) if i != beam]
    frac = ((X - 117) / A0 + .318) % 1
    metal = species != 'O'
    odd = np.all(np.abs(frac * 8 - np.rint(frac * 8)) < 1e-7, axis=1) & np.all(np.rint(frac * 8).astype(int) % 2 == 1, axis=1)
    Td = aggregate_columns(X, Y, species, region, metal & odd, image_axes, 'Td')
    Oh = aggregate_columns(X, Y, species, region, metal & ~odd, image_axes, 'Oh')
    itd = {tuple(n): i for i, n in enumerate(Td['labels'])}
    ioh = {tuple(n): i for i, n in enumerate(Oh['labels'])}
    lo, hi = Td['labels'].min(axis=0), Td['labels'].max(axis=0)
    centers, oh_rows, td_rows = [], [], []
    # Empty checkerboard holes are offset a0/4 from the Td projected sublattice.
    for h in range(lo[0] + 1, hi[0]):
        for v in range(lo[1] + 1, hi[1]):
            if (h + v) % 2 != 1:
                continue
            ot = [(h, v), (h + 1, v), (h, v + 1), (h + 1, v + 1)]
            tt = [(h - 1, v), (h + 1, v), (h, v - 1), (h, v + 1)]
            if all(n in ioh for n in ot) and all(n in itd for n in tt):
                centers.append([h, v])
                oh_rows.append([ioh[n] for n in ot])
                td_rows.append([itd[n] for n in tt])
    labels = np.array(centers)
    io, it = np.array(oh_rows), np.array(td_rows)
    pos = Oh['pos'][io].mean(axis=1)
    core_pos = Oh['core_pos'][io].mean(axis=1)
    td_vertex = Td['pos'][it]
    dh = np.linalg.norm(td_vertex[:, 1] - td_vertex[:, 0], axis=1)
    dv = np.linalg.norm(td_vertex[:, 3] - td_vertex[:, 2], axis=1)
    q0 = ((labels / 4 + .125) - .318) * A0
    phase_region = np.where(np.max(np.abs(q0), axis=1) <= 55, 1,
                            np.where(np.max(np.abs(q0), axis=1) <= 85, 2, 3))
    return dict(labels=labels, positions=pos, core_only_positions=core_pos,
                topology_positions=q0, ratio_raw=dh / dv,
                td_dh_A=dh, td_dv_A=dv,
                column_spread_A=np.sqrt(np.mean(Oh['spread'][io]**2, axis=1)),
                core_fraction=Oh['core_fraction'][io].mean(axis=1),
                beam_overlap_fraction=np.maximum(Oh['front_fraction'][io].max(axis=1), Td['front_fraction'][it].max(axis=1)),
                column_atoms_mean=Oh['count'][io].mean(axis=1), region=phase_region,
                td_proxy_positions=td_vertex.mean(axis=1),
                Td_columns=len(Td['labels']), Oh_columns=len(Oh['labels']), image_axes=image_axes)


def reference_fit(labels, positions, core_mask):
    design = np.column_stack([np.ones(len(labels)), labels])
    coefficients = np.linalg.lstsq(design[core_mask], positions[core_mask], rcond=None)[0]
    reference = design @ coefficients
    # Horizontal/vertical dense grid steps; primitive diagonal lattice basis has positive determinant.
    B_hv = coefficients[1:].T
    B = B_hv @ np.array([[1, -1], [1, 1]])
    assert np.linalg.det(B) > 0
    return reference, B, B_hv, coefficients


def gradient_fit(reference, positions, B, sigma=1.5, query=None):
    """Gaussian local displacement derivative with an intercept and physical Cartesian axes."""
    query = reference if query is None else query
    invB = np.linalg.inv(B)
    uv = reference @ invB.T
    quv = query @ invB.T
    tree = cKDTree(uv)
    neighborhoods = tree.query_ball_point(quv, 3.5 * sigma)
    disp = positions - reference
    G = np.full((len(query), 2, 2), np.nan)
    smoothed_disp = np.full((len(query), 2), np.nan)
    rms = np.full(len(query), np.nan)
    effective_n = np.zeros(len(query))
    for i, ids in enumerate(neighborhoods):
        if len(ids) < 6:
            continue
        duv = uv[ids] - quv[i]
        w = np.exp(-np.sum(duv**2, axis=1) / (2 * sigma**2))
        design = np.column_stack([np.ones(len(ids)), reference[ids] - query[i]])
        a = design * np.sqrt(w[:, None])
        b = disp[ids] * np.sqrt(w[:, None])
        co, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
        if rank < 3:
            continue
        G[i] = co[1:].T
        smoothed_disp[i] = co[0]
        rms[i] = np.sqrt(np.sum(w * np.sum((disp[ids] - design @ co)**2, axis=1)) / w.sum())
        effective_n[i] = w.sum()**2 / (w @ w)
    J = G + np.eye(2)
    valid = np.isfinite(G).all(axis=(1, 2)) & (np.linalg.det(J) > 0)
    return dict(G=G, valid=valid, displacement=disp, smooth_displacement=smoothed_disp,
                eps_xx=G[:, 0, 0], eps_yy=G[:, 1, 1], eps_xy=(G[:, 0, 1] + G[:, 1, 0]) / 2,
                rotation_deg=np.degrees((G[:, 1, 0] - G[:, 0, 1]) / 2),
                derivative_fit_RMS_A=rms, effective_neighbors=effective_n)


def summarize(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)),
                min=float(x.min()), max=float(x.max()))


def analytic_checks():
    labels = np.array(list(itertools.product(range(-8, 9), repeat=2)))
    B = np.array([[2., -.1], [.2, 2.1]])
    R = labels @ B.T
    keep = ~((labels[:, 0] > 4) & (labels[:, 1] > 3))
    R = R[keep]
    errors = {}
    phi = .07
    fields = {'translation': np.eye(2), 'normal': np.diag([1.12, .94]),
              'symmetric_shear': np.array([[1, .02], [.02, 1]]),
              'simple_shear': np.array([[1, .04], [0, 1]]),
              'finite_rotation_infinitesimal_convention': np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])}
    for name, F in fields.items():
        P = R @ F.T + [3., -7.]
        result = gradient_fit(R, P, B)
        error = float(np.nanmax(np.abs(result['G'] - (F - np.eye(2)))))
        assert error < 2e-13
        errors[name] = error
    n = labels[keep]
    P = R @ fields['simple_shear'].T + [7, 5]
    ref, bf, _, _ = reference_fit(n, P, np.max(abs(n), axis=1) <= 4)
    errors['fresh_core_reference_global_affine_zero'] = float(np.nanmax(abs(gradient_fit(ref, P, bf)['G'])))
    assert errors['fresh_core_reference_global_affine_zero'] < 1e-13
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--initial', type=Path, default=INITIAL)
    parser.add_argument('--final', type=Path, default=FINAL)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    X, sp, region, initial_info = read_xyz(args.initial)
    Y, final_sp, final_region, final_info = read_xyz(args.final)
    assert X.shape == Y.shape
    assert np.array_equal(sp, final_sp) and np.array_equal(region, final_region), 'Same-row species/region correspondence is required'
    d = projected_centers(X, Y, sp, region, args.beam)
    labels, positions = d['labels'], d['positions']
    # Distance to final lateral Mn cations, an operational interface-clearance proxy.
    # Front/back grains necessarily overlap the core; they are not the lateral boundary.
    absq = abs(X - 117)
    lateral_mn = (sp == 'Mn') & (absq[:, args.beam] <= np.max(absq[:, d['image_axes']], axis=1) + 1e-7)
    clearance = cKDTree(Y[lateral_mn][:, d['image_axes']]).query(positions)[0]
    spacing_guess = A0 / (2 * np.sqrt(2))
    core_mask = (clearance >= 4 * spacing_guess) & (d['core_fraction'] >= .5) & (d['region'] == 1)
    ref, B, B_hv, coeff = reference_fit(labels, positions, core_mask)
    spacing = float(np.mean(np.linalg.norm(B, axis=0)))
    # Re-select with measured final-core spacing rather than the starting estimate.
    core_mask = (clearance >= 4 * spacing) & (d['core_fraction'] >= .5) & (d['region'] == 1)
    ref, B, B_hv, coeff = reference_fit(labels, positions, core_mask)
    spacing = float(np.mean(np.linalg.norm(B, axis=0)))
    g = gradient_fit(ref, positions, B, args.sigma)
    J = g['G'] + np.eye(2)
    # Correct dense diagonal geometry using measured core horizontal/vertical ratio; not an experimental scan correction.
    core_ratio = float(np.median(d['ratio_raw'][core_mask]))
    ratio = d['ratio_raw'] / core_ratio
    folded = np.maximum(ratio, 1 / ratio)
    area = 8 * np.abs(np.linalg.det(J @ B)) / 100
    aspect_smooth = np.linalg.norm(J @ B_hv[:, 0], axis=1) / np.linalg.norm(J @ B_hv[:, 1], axis=1)
    aspect_smooth /= np.linalg.norm(B_hv[:, 0]) / np.linalg.norm(B_hv[:, 1])
    roi = (d['region'] != 1) & (d['beam_overlap_fraction'] < 1e-8) & g['valid']
    display_center = (np.r_[1, 4 * (.318 - .125), 4 * (.318 - .125)] @ coeff)
    data = {k: d[k] for k in ['labels', 'positions', 'region', 'column_spread_A', 'beam_overlap_fraction', 'column_atoms_mean', 'core_fraction', 'topology_positions']}
    data.update({k: v for k, v in g.items() if k != 'G'})
    data.update(reference=ref, ratio_hv=ratio, ratio_folded=folded, ratio_hv_raw=d['ratio_raw'],
                ratio_hv_smoothed=aspect_smooth, area_nm2=area, roi_lateral=roi,
                core_reference=core_mask, core_interface_clearance_A=clearance,
                sigma_A=args.sigma * spacing, center_2d=display_center,
                B_primitive=B, B_dense_hv=B_hv, G=g['G'], beam_axis=args.beam)
    np.savez_compressed(OUT / 'paper_projection.npz', **data)
    print(f'Projected cells {len(labels)}, reference cells {core_mask.sum()}, lateral ROI {roi.sum()}', flush=True)
    report = {
        'method_label': 'Paper-reference structural analysis from XYZ full-thickness projection; coordinate-derived lattice centers replace HAADF intensity fitting.',
        'not_a_STEM_simulation': True,
        'experimental_steps_replaced': ['No multislice/detector simulation or image low-pass filter.', 'No negative-Gaussian fit to HAADF valleys; empty-hole centers estimated from four surrounding projected octahedral cation columns.', 'Fig3 diamond geometry from four tetrahedral projected columns; no image-template detection or polynomial line-intersection refinement.'],
        'source': {'main': 'Nature DOI 10.1038/s41586-019-1899-3, Fig3/4', 'SI': 'Methods printed pp3-4; S8 printed p16; S16 p24; S17 p25'},
        'inputs': {'initial': initial_info, 'final': final_info},
        'reference': 'Final projected central Co-core region, >=4 projected primitive lattice spacings from nearest final lateral Mn cation and Co column fraction >=0.5. All fitted coordinates from final XYZ; initial XYZ supplies only discrete topology and geometric region identities. DFT not used.',
        'reference_projection_caveat': 'Full-depth projected core columns also include front/back Mn cations, as in a projection. Main core calibration uses these projected markers in the core-dominant region; pure-Co-only reference is checked as sensitivity.',
        'projection': {'beam_parent_axis': ['[100]', '[010]', '[001]'][args.beam], 'image_horizontal_parent_axis': ['[100]', '[010]', '[001]'][d['image_axes'][0]], 'image_vertical_parent_axis': ['[100]', '[010]', '[001]'][d['image_axes'][1]], 'depth': 'all atoms, no slab'},
        'reference_basis_primitive_columns_A': B.tolist(), 'reference_dense_step_columns_A': B_hv.tolist(),
        'reference_coefficients_r0_h_v': coeff.tolist(), 'core_reference_cell_count': int(core_mask.sum()),
        'primitive_spacing_A': spacing, 'sigma_primitive_cells': args.sigma, 'sigma_A': args.sigma * spacing,
        'core_clearance_min_A': float(clearance[core_mask].min()),
        'core_clearance_min_primitive_cells': float(clearance[core_mask].min() / spacing),
        'kernel_convention_evidence': 'Primitive spacing a/(2sqrt2) agrees with measured Fig4b displacement marker spacing~0.287 nm and stated~0.5 nm resolution. The paper does not explicitly give the sampling-cell-to-conventional-cell conversion; this is an inferred crystallographic correspondence.',
        'area_convention': 'Conventional-cell-equivalent projected area =8*primitive local cell area. Factor8 follows spinel projection and Fig4 conventional Co area~0.65 nm²; not a 3D volume.',
        'strain_convention': 'Gaussian weighted local linear fit to displacement p-r_ref, including intercept; epsxx=Gxx, epsyy=Gyy, epsxy=(Gxy+Gyx)/2, theta=(Gyx-Gxy)/2 rad. Rotation displayed in degrees, positive counterclockwise with x right and y up.',
        'smoothing': 'Gaussian sigma1.5 primitive cells, radial support3.5sigma. Local linear regression is the selected numerical differentiation implementation; author MATLAB source unavailable.',
        'Fig3_ratio': 'dh/dv from lengths of horizontal and vertical diamond diagonals joining Td column centroids, divided by median ratio in deep core. Histogram=max(r,1/r). Equal weight per projected cell. No experimental numerical scan-drift factors applied.',
        'Fig3_core_ratio_calibration': core_ratio,
        'lateral_ROI': 'Non-core cell centers with zero beam-normal-grain contribution in every constituent Oh and Td column. Gaussian derivative neighborhoods may cross into excluded overlap regions. Beam-normal sectors inferred geometrically from initial particle faces. Strain remains an apparent projected field; overlapping side grains near GB are not removed.',
        'atom_identity': 'Original XYZ same-row correspondence assumed for topology; full species/region array equality checked in this run. No atom IDs are available to prove identity after same-species reordering.',
        'column_definition': 'Equal-weight centroid of final cation positions sharing one initial projected crystallographic column. Spreading/splitting through depth quantified; no HAADF Z weighting assumed.',
        'number_of_cells': len(labels), 'valid_cells': int(g['valid'].sum()), 'lateral_ROI_cells': int(roi.sum()),
        'projected_Td_columns': d['Td_columns'], 'projected_Oh_columns': d['Oh_columns'],
        'analytic_checks_max_gradient_error': analytic_checks(), 'statistics': {}, 'sensitivity': {}
    }
    for name, m in [('reference_core', core_mask), ('all_projected_core', d['region'] == 1), ('lateral_CL', roi & (d['region'] == 2)), ('lateral_island', roi & (d['region'] == 3)), ('beam_overlap_shell', (d['region'] != 1) & ~roi)]:
        m = m & g['valid']
        report['statistics'][name] = {k: summarize(data[k][m]) for k in ['eps_xx', 'eps_yy', 'eps_xy', 'rotation_deg', 'ratio_hv', 'ratio_folded', 'area_nm2', 'column_spread_A']}
    for sigma in [1., 2., 1.5 * 2 * np.sqrt(2)]:
        alt = gradient_fit(ref, positions, B, sigma)
        report['sensitivity'][f'sigma_{sigma:.5g}_primitive_cells'] = {k: {'median_abs_difference_percentage_points': float(np.nanmedian(abs(alt[k][roi] - g[k][roi])) * 100), 'p90_abs_difference_percentage_points': float(np.nanquantile(abs(alt[k][roi] - g[k][roi]), .9) * 100)} for k in ['eps_xx', 'eps_yy', 'eps_xy']}
    for margin in [8, 12]:
        mask = (clearance >= margin * spacing) & (d['core_fraction'] >= .5) & (d['region'] == 1)
        rr, bb, _, _ = reference_fit(labels, positions, mask)
        alt = gradient_fit(rr, positions, bb, args.sigma)
        report['sensitivity'][f'core_margin_{margin}_primitive_cells'] = {'reference_n': int(mask.sum()), 'basis_A': bb.tolist(), **{k: float(np.nanmedian(abs(alt[k][roi] - g[k][roi])) * 100) for k in ['eps_xx', 'eps_yy']}}
    pure_mask = core_mask & np.isfinite(d['core_only_positions']).all(axis=1)
    rp, bp, _, _ = reference_fit(labels, d['core_only_positions'], pure_mask)
    alt = gradient_fit(rp, positions, bp, args.sigma)
    report['sensitivity']['Co_only_reference'] = {'basis_A': bp.tolist(), **{k: float(np.nanmedian(abs(alt[k][roi] - g[k][roi])) * 100) for k in ['eps_xx', 'eps_yy']}}
    # Compare the centroid-of-Td-diamond proxy to the Oh empty-hole proxy.
    rt, bt, _, _ = reference_fit(labels, d['td_proxy_positions'], core_mask)
    alt = gradient_fit(rt, d['td_proxy_positions'], bt, args.sigma)
    report['sensitivity']['Td_diamond_center_proxy'] = {k: float(np.nanmedian(abs(alt[k][roi] - g[k][roi])) * 100) for k in ['eps_xx', 'eps_yy', 'eps_xy']}
    fields = ['eps_xx', 'eps_yy', 'eps_xy', 'rotation_deg', 'ratio_hv', 'ratio_folded', 'area_nm2', 'column_spread_A', 'beam_overlap_fraction']
    with (OUT / 'projected_cells.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell_id', 'h', 'v', 'x_A', 'y_A', 'reference_x_A', 'reference_y_A', 'region', 'core_reference', 'lateral_ROI', 'valid'] + fields)
        for i in range(len(labels)):
            w.writerow([i, *labels[i], *positions[i], *ref[i], d['region'][i], int(core_mask[i]), int(roi[i]), int(g['valid'][i])] + [data[k][i] for k in fields])
    # L-shaped image upper-right corner paths, on the reference lattice, no sampling from an arbitrary thin slab.
    trace_rows = []
    ref_tree = cKDTree(ref)
    for offset in [1, 3, 5]:
        distance = np.linspace(-8, 8, 65)
        q = np.tile([55 + offset * spacing, 55 + offset * spacing], (len(distance), 1))
        q[distance <= 0, 0] += distance[distance <= 0] * spacing
        q[distance > 0, 1] -= distance[distance > 0] * spacing
        lv = ((q / A0 + .318) - .125) * 4
        rq = np.column_stack([np.ones(len(q)), lv]) @ coeff
        tg = gradient_fit(ref, positions, B, args.sigma, query=rq)
        tj = tg['G'] + np.eye(2)
        ratio_t = np.linalg.norm(tj @ B_hv[:, 0], axis=1) / np.linalg.norm(tj @ B_hv[:, 1], axis=1)
        ratio_t /= np.linalg.norm(B_hv[:, 0]) / np.linalg.norm(B_hv[:, 1])
        area_t = 8 * np.abs(np.linalg.det(tj @ B)) / 100
        nd, ni = ref_tree.query(rq)
        for j, s in enumerate(distance):
            trace_rows.append(dict(offset_cells=offset, distance_cells=float(s), reference_x_A=float(rq[j, 0]), reference_y_A=float(rq[j, 1]),
                eps_xx_percent=float(tg['eps_xx'][j] * 100), eps_yy_percent=float(tg['eps_yy'][j] * 100), eps_xy_percent=float(tg['eps_xy'][j] * 100),
                rotation_deg=float(tg['rotation_deg'][j]), ratio_hv=float(ratio_t[j]), area_nm2=float(area_t[j]), roi_lateral=bool(roi[ni[j]] and nd[j] < spacing)))
    with (OUT / 'corner_line_traces.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0])); w.writeheader(); w.writerows(trace_rows)
    report['line_traces'] = 'Upper-right image corner L paths at 1,3,5 primitive-cell spacings outside nominal core faces, -8..8 primitive-cell arc distance about each path bend. Gaussian smoothed ratio from Jacobian; not raw four-Td ratio. Overlap flags supplied in CSV.'
    report['corner_trace_samples'] = [r for r in trace_rows if r['offset_cells'] == 3 and r['distance_cells'] in [-6., 0., 6.]]
    (OUT / 'paper_method_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {q: r[q]['median'] for q in ['eps_xx', 'eps_yy', 'rotation_deg', 'ratio_folded']} for k, r in report['statistics'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
