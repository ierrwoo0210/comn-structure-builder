"""Auditable Fig.4 reconstruction on actual image-fitted lattice sites.

Author raw images/MATLAB are unavailable. Explicit choices that are not fixed by
the article are recorded in revision_method.json. No experimental data invented.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.tools/python_packages'),str(ROOT/'.tools/plot_packages')]
import numpy as np
from analyze_gaussian_image_strain import compute_result, save_csv, supported_gradient
from analyze_paper_strain import gradient_fit, reference_fit, analytic_checks

SOURCE=ROOT/'output/strain_gaussian_image'
OUT=SOURCE/'fig4_revision'


def read_cell(path):
    lines=path.read_text(encoding='utf-8').splitlines()
    scale=float(lines[1]); assert scale>0
    return scale*np.array([[float(v) for v in x.split()[:3]] for x in lines[2:5]])


def projected_mn(cell):
    a,b,c=cell
    beam=(a+b)/np.linalg.norm(a+b)
    basal=a-b; basal-=np.dot(basal,beam)*beam
    long=c-np.dot(c,beam)*beam
    return dict(a_sqrt2_A=float(np.linalg.norm(basal)),c_A=float(np.linalg.norm(long)),
                projected_area_nm2=float(np.linalg.norm(np.cross(basal,long))/100),
                conventional_vectors_A=cell.tolist(),
                interpretation='Beam parallel to a+b; basal vector a-b and c projected perpendicular to beam. a*sqrt(2) is exact for tetragonal a=b, gamma=90; actual DFT vectors retained.')


def reference_lines(d,mn):
    C=4*d['B_dense_hv']
    ax,ay=np.linalg.norm(C,axis=0)
    short,long=mn['a_sqrt2_A'],mn['c_A']
    return dict(eps_xx=dict(co=0.,mn_a_sqrt2=100*(short/ax-1),mn_c=100*(long/ax-1)),
                eps_yy=dict(co=0.,mn_a_sqrt2=100*(short/ay-1),mn_c=100*(long/ay-1)),
                aspect=dict(co=float(ay/ax),mn_high=long/short,mn_low=short/long),
                area_nm2=dict(co=float(abs(np.linalg.det(C))/100),mn=mn['projected_area_nm2']))


def make_rows(d,anchor,offsets):
    index={tuple(label):i for i,label in enumerate(d['labels'])}
    H=d['B_dense_hv']; spacing=2*np.linalg.norm(H,axis=0)/10
    rows=[]
    for k in offsets:
        bend=anchor+2*k
        for s in range(-8,9):
            label=bend+(np.array([2*s,0]) if s<=0 else np.array([0,-2*s]))
            cid=index.get(tuple(label),-1)
            base=dict(path_id='core' if k<0 else f'+{k}',offset_cells=k,distance_cells=s,
                      signed_reference_distance_nm=float(s*spacing[0 if s<=0 else 1]),
                      cell_id=cid,h=int(label[0]),v=int(label[1]))
            if cid<0:
                base.update(x_nm=np.nan,y_nm=np.nan,eps_xx_percent=np.nan,eps_yy_percent=np.nan,
                    eps_xy_percent=np.nan,rotation_deg=np.nan,aspect_vh=np.nan,area_nm2=np.nan,
                    fit_valid=False,valid=False,Gxx=np.nan,Gxy=np.nan,Gyx=np.nan,Gyy=np.nan,
                    region=0,core_reference=False,fit_nrmse=np.nan)
            else:
                valid=bool(d['valid'][cid]); fit=bool(d['fit_valid'][cid])
                xy=(d['positions'][cid]-d['center_2d'])/10
                G=d['G'][cid]; J=np.eye(2)+G
                edges=J@(4*H)
                aspect=np.linalg.norm(edges[:,1])/np.linalg.norm(edges[:,0])
                base.update(x_nm=float(xy[0]) if fit else np.nan,y_nm=float(xy[1]) if fit else np.nan,
                    eps_xx_percent=float(100*G[0,0]) if valid else np.nan,
                    eps_yy_percent=float(100*G[1,1]) if valid else np.nan,
                    eps_xy_percent=float(50*(G[0,1]+G[1,0])) if valid else np.nan,
                    rotation_deg=float(np.degrees((G[1,0]-G[0,1])/2)) if valid else np.nan,
                    aspect_vh=float(aspect) if valid else np.nan,
                    area_nm2=float(np.linalg.det(edges)/100) if valid else np.nan,
                    fit_valid=fit,valid=valid,Gxx=float(G[0,0]),Gxy=float(G[0,1]),
                    Gyx=float(G[1,0]),Gyy=float(G[1,1]),region=int(d['region'][cid]),
                    core_reference=bool(d['core_reference'][cid]),fit_nrmse=float(d['fit_nrmse'][cid]))
            rows.append(base)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    old=dict(np.load(SOURCE/'gaussian_strain.npz'))
    fit=dict(np.load(SOURCE/'valley_fits_s060.npz'))
    shift_from_proxy=np.linalg.norm(old['positions']-old['seed_positions'],axis=1)
    clearance_lower_bound=old['core_interface_clearance_A']-shift_from_proxy
    # Use a deliberately more interior reference than any of the plausible unit
    # conventions in the article. 4 conventional Co cells is >4 projected repeats.
    margin=4*float(np.max(np.linalg.norm(4*old['B_dense_hv'],axis=0)))
    core=old['core_reference'] & (clearance_lower_bound>=margin)
    for _ in range(3):
        _,_,H,_=reference_fit(old['labels'],old['positions'],core)
        margin=4*float(np.max(np.linalg.norm(4*H,axis=0)))
        core=old['core_reference'] & (clearance_lower_bound>=margin)
    assert core.sum()>=100
    d=compute_result(old,fit,core)
    d['seed_positions']=old['seed_positions'].copy()
    C=4*d['B_dense_hv']
    # Store raw directional aspect, matching Fig4e; not core-normalized Fig3 ratio.
    d['aspect_vh']=np.linalg.norm((np.eye(2)+d['G'])@C[:,1],axis=1)/np.linalg.norm((np.eye(2)+d['G'])@C[:,0],axis=1)
    np.savez_compressed(OUT/'revised_data.npz',**d)
    # Closest existing core site to the nominal top/right core corner. Resolve the
    # symmetric tie lexicographically by larger h, keeping every choice explicit.
    candidates=np.flatnonzero((d['region']==1)&d['fit_valid'])
    distance=np.linalg.norm(d['topology_positions'][candidates]-[55.,55.],axis=1)
    ties=candidates[np.isclose(distance,distance.min(),atol=1e-8)]
    anchor_id=max(ties,key=lambda i:tuple(d['labels'][i]))
    anchor=d['labels'][anchor_id]
    rows=make_rows(d,anchor,[-3,*range(1,8)])
    save_csv(OUT/'profiles.csv',rows)
    assert len(rows)==136 and all(r['cell_id']>=0 for r in rows)
    assert all(r['fit_valid'] and r['valid'] for r in rows)
    assert all(r['region']==1 for r in rows if r['path_id']=='core')
    for r in rows:
        i=r['cell_id'];assert np.array_equal(d['labels'][i],[r['h'],r['v']])
        assert np.allclose(np.array([r['x_nm'],r['y_nm']])*10+d['center_2d'],d['positions'][i],atol=1e-10)
        assert abs(r['eps_xx_percent']-d['eps_xx'][i]*100)<1e-10
        assert abs(r['area_nm2']-8*np.linalg.det((np.eye(2)+d['G'][i])@d['B_primitive'])/100)<1e-10
    desktop=Path('C:/Users/riphy/OneDrive/바탕 화면')
    mn_dft=projected_mn(read_cell(desktop/'CONTCAR_Mn3O4'))
    mn_lit=projected_mn(np.diag([5.765,5.765,9.442]))
    ref=dict(dft=reference_lines(d,mn_dft),literature=reference_lines(d,mn_lit),
             dft_bulk=mn_dft,literature_bulk=mn_lit,
             core_x_A=float(np.linalg.norm(C[:,0])),core_y_A=float(np.linalg.norm(C[:,1])),
             core_area_nm2=float(np.linalg.det(C)/100),
             horizontal_repeat_nm=float(2*np.linalg.norm(d['B_dense_hv'][:,0])/10),
             vertical_repeat_nm=float(2*np.linalg.norm(d['B_dense_hv'][:,1])/10),
             source='Oh et al. Nature577,359-363(2020), DOI10.1038/s41586-019-1899-3; user CONTCAR_Mn3O4',
             note='Both line sets use THIS image-fitted Co reference. Literature version uses only paper bulk Mn constants, not paper normalized numeric strain lines.')
    (OUT/'reference_lines.json').write_text(json.dumps(ref,indent=2),encoding='utf8')
    # Quantify the unresolved kernel-unit convention; do not fabricate confidence
    # bounds. Alternative has physical sigma=1.5 axial repeat (sqrt2 broader).
    mask=d['fit_valid'];ids=np.array([r['cell_id'] for r in rows])
    alternative=gradient_fit(d['reference'][mask],d['positions'][mask],d['B_primitive'],
                             sigma=1.5*np.sqrt(2),query=d['reference'][ids])
    sensitivity=[]
    for key in ['eps_xx','eps_yy','eps_xy']:
        values=100*(alternative[key]-d[key][ids])
        sensitivity.append(dict(component=key,median_abs_difference_pp=float(np.median(abs(values))),
                                max_abs_difference_pp=float(np.max(abs(values)))))
    save_csv(OUT/'kernel_convention_sensitivity.csv',sensitivity)
    # Full conventional cells whose four vertices are actual fitted valley sites.
    # These are an audit companion, separate from 6x display glyphs.
    index={tuple(v):i for i,v in enumerate(d['labels'])};corners=[]
    for r in rows:
        if r['distance_cells'] not in [-6,0,6]:continue
        labels=np.array([r['h'],r['v']])+np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
        vertices=[index.get(tuple(v),-1) for v in labels]
        valid=all(i>=0 and d['fit_valid'][i] for i in vertices)
        corners.append(dict(path_id=r['path_id'],distance_cells=r['distance_cells'],center_cell_id=r['cell_id'],
                            vertex_cell_ids=vertices,valid=valid,
                            vertices_nm=((d['positions'][vertices]-d['center_2d'])/10).tolist() if valid else None))
    (OUT/'actual_cell_vertices.json').write_text(json.dumps(corners,indent=2),encoding='utf8')
    method=dict(
        status='Source-grounded reconstruction on a Gaussian projected image; not an exact author-code or experimental reproduction.',
        source_npz_sha256=hashlib.sha256((SOURCE/'gaussian_strain.npz').read_bytes()).hexdigest(),
        mn_dft_source_path=str(desktop/'CONTCAR_Mn3O4'),
        mn_dft_sha256=hashlib.sha256((desktop/'CONTCAR_Mn3O4').read_bytes()).hexdigest(),
        source_image='images.npz:image_s060; all individual atoms projected [100]; Gaussian0.60A plus lowpass0.25A; existing fitted centers reused.',
        reference=dict(n=int(core.sum()),minimum_proxy_interface_clearance_nm=float(d['core_interface_clearance_A'][core].min()/10),
                       minimum_fitted_site_clearance_lower_bound_nm=float(clearance_lower_bound[core].min()/10),
                       required_clearance_nm=margin/10,
                       rule='More conservative than SI: >=4 conventional Co cell lengths from final lateral Mn cations. Stored distance is final coordinate-proxy to Mn; subtract proxy-to-imagefit shift by triangle inequality to bound fitted-site clearance. All reference sites accepted in every prior Gaussian case.',
                       center_2d_A=d['center_2d'].tolist(),conventional_basis_A=C.tolist()),
        anchor=dict(cell_id=int(anchor_id),label=anchor.tolist(),initial_xy_A=d['topology_positions'][anchor_id].tolist(),
                    nominal_corner_A=[55.,55.],rule='Nearest fitted lattice site INSIDE projected Co core; tie resolved by larger h. Not the exact geometric interface corner.'),
        paths='Core -3 and shell +1..+7 relative to the chosen core-corner site. Bend label = anchor+(2k,2k). s<=0: bend+(2s,0); s>0: bend+(0,-2s). Every plotted point uses its actual image-fitted center and site derivative.',
        distance='Signed integer horizontal/vertical valley-repeat count; unit vectors2H. NOT diagonal nearest-neighbor |B|, and NOT conventional4H. This is the explicit reconstruction choice consistent with Fig4 scale; author exact basis not published.',
        derivative=dict(formula='u=measured-reference; eps=sym(grad u); theta=(G_yx-G_xy)/2',
                        sigma_primitive_cells=1.5,sigma_physical_nm=float(1.5*np.mean(np.linalg.norm(d['B_primitive'],axis=0))/10),
                        note='2D primitive valley basis B=[H_x+H_y,-H_x+H_y]. Path unit2H differs; SI does not disclose numeric basis. Alternative axial-width sensitivity exported.',
                        sensitivity=sensitivity),
        aspect='Raw norm((I+G)C_y)/norm((I+G)C_x); no normalization to core and no max(r,1/r).',
        area='det((I+G)C)/100 nm2 where C=4H; conventional-cell-equivalent projected area.',
        reference_lines='DFT default, literature alternate; both converted against current core x/y metric. No pasted0.85/16.8% constants.',
        white_glyphs='Display only centered at actual fitted sites. SI specifies strain exaggeration6. Selected explicit implementation: I+6sym(G)+skew(G), baseline conventional cell scaled0.20; rotation not exaggerated. Author exact glyph algorithm unavailable.',
        actual_cells='actual_cell_vertices.json contains quadrilaterals formed by four measured sites at h/v+-2; no glyph distortion applied.',
        verification=dict(profile_rows=len(rows),valid_rows=sum(r['valid'] for r in rows),
                          exact_existing_lattice_ids=True,profile_positions_match_fits=True,
                          profiles_match_site_strain=True,area_basis_identity=True,
                          analytic_gradient_max_error=analytic_checks()),
        limits=['Gaussian projection is not multislice HAADF; experimental scan distortions are absent.',
                'Front/back Mn signal remains in projected core. Core reference is not pure-element image.',
                'Paths chosen for this particle; not the coordinates of the experimental particle.',
                'Raw author images and MATLAB not available; exact path basis, derivative discretization and glyph formula cannot be certified identical.'])
    (OUT/'revision_method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2),encoding='utf8')
    notes=f'''Fig. 4d/e 재구성: 계산 정의와 원문 대조

이 폴더의 결과는 final_11nm.xyz에서 만든 Gaussian 투영 영상을 분석한 결과입니다.
논문의 실험 입자 데이터나 HAADF multislice 영상이 아닙니다. 원저자 raw 영상 및 MATLAB
코드를 확보하지 못했으므로 "완전히 동일한 원저자 구현"으로 주장하지 않습니다.
기존 결과 파일은 보존했고, 이 폴더에 새 기준과 실제 격자점 경로로 다시 계산했습니다.

1. 논문에 명시된 절차
Oh et al., Nature 577, 359-363 (2020), DOI 10.1038/s41586-019-1899-3.
SI Methods 인쇄 3-4쪽: low-pass 영상 → 음의 2D Gaussian으로 valley 중심 피팅 →
격자 ID → Co 내부 기준 격자 → 변위 미분. 기준은 계면에서 4 unit cells 이상 내부,
Gaussian 미분 폭은 reference lattice의 1.5 unit cells.
Fig. 4d/e: corner 기준의 부호 있는 경로 위치, εxx, εyy, 종횡비, 투영 셀 면적.
SI Fig. S11 인쇄 19쪽: 흰색 셀 도식의 strain은 6배 과장.
논문은 u,v의 구체적 수치/방향, 경로 offset 값, 흰 도식의 상세 생성 코드를 공개하지 않음.

2. 이번 수정에서 바뀐 것
• 이전의 임의 연속좌표 129점/경로를 폐기하고 실제 피팅된 valley 격자점 17개/경로 사용.
  코어 경로 1개와 +1~+7 경로를 합해 {len(rows)}개이며 모두 피팅 및 미분 검사를 통과.
  곡선의 점을 추가 보간하거나 사후 smoothing하지 않음. 점 사이 연결선은 시각적 연결.
• 거리 1칸 = 같은 가로/세로 행의 이웃 valley까지의 반복. h 또는 v label을 2씩 이동.
  가로 {ref['horizontal_repeat_nm']:.6f} nm, 세로 {ref['vertical_repeat_nm']:.6f} nm.
  기존 대각 primitive 간격 약0.2893 nm나 conventional 격자상수 약0.818 nm와 구별.
  논문 그림 크기와 일관되는 재구성 선택이며, 저자의 정확한 거리 단위와 동일하다고
  확정할 raw 자료는 없음. 축의 unit cells는 본 그림에서 이 투영 반복 셀을 뜻함.
• 경로 0은 각 경로의 corner 격자점. 음수는 왼쪽 팔, 양수는 아래쪽 팔.
  경로 이름 +1~+7은 core corner 격자점에서 양 축 방향으로 더한 반복 횟수.
  이 offset은 core-shell 두께를 법선 방향으로 직접 측정한 값이 아님.
• 기준 anchor: 실제 Co 내부 격자점 ID {anchor_id}, label {anchor.tolist()}.
  초기 구조 위치 {d['topology_positions'][anchor_id].tolist()} Å. 기하학적 (55,55) Å
  모서리 그 자체가 아니라 그에 가장 가까운 Co 내부 피팅 격자점.
  코어 비교 경로는 anchor에서 -3 반복 안쪽. 기존 -4 연속좌표 경로와 위치가 다름.
• 코어 기준을 더 보수적으로 선택: conventional Co 셀 4개 길이 이상 내부.
  {int(core.sum())}개. 기존 최종 좌표 proxy에서 최종 측면 Mn까지의 거리에서 피팅 이동량을
  빼서 얻은 실제 피팅점 이격의 하한은 {method['reference']['minimum_fitted_site_clearance_lower_bound_nm']:.4f} nm.
  앞뒤 shell이 아니라 측면 Mn 원자를 기준으로 한 보수적인 거리 하한임.
  이 조건으로 기준 격자와 전체 strain을 재피팅했으므로 이전 곡선값과 달라질 수 있음.
• 종횡비는 raw dv/dh. 코어 비율로 추가 정규화하거나 큰 값/작은 값으로 접지 않음.
• 면적은 conventional-cell-equivalent 2D 면적 det[(I+G)·4H]/100 nm². 3D 부피 아님.

3. 흰 사각형과 실제 unit cell의 구별
Fig.4처럼 흰 도식은 변형을 강조하기 위한 glyph임. 이번 구현은
I + 6·sym(G) + skew(G): strain 6배, 회전 1배. 기본 conventional 셀 변 길이는
표시용으로 0.20배. 도식 중심은 반드시 실제 피팅 격자점에 놓음.
흰 도식의 크기는 실제 unit-cell 면적을 뜻하지 않음. 그래프 면적은 과장하지 않음.
원문은 6배 strain만 명시하므로 이 상세 구현은 재구성 선택임.
actual_cell_vertices.json은 별도로 h/v ±2의 실제 측정된 네 격자점으로 만든
conventional 셀 꼭짓점 좌표를 제공함. 해당 좌표에는 표시용 과장이 없음.

4. Mn 기준 점선
주 결과 dft: 제공된 CONTCAR_Mn3O4의 실제 격자벡터 사용.
beam ∥ a+b, basal a-b와 c를 beam에 수직으로 투영.
a√2 표기는 정방정계에서 정확하며, DFT의 작은 비정방 왜곡은 실제 벡터로 유지.
투영 basal={mn_dft['a_sqrt2_A']:.6f} Å, c={mn_dft['c_A']:.6f} Å.
비교선은 새 영상 코어 x={ref['core_x_A']:.6f} Å, y={ref['core_y_A']:.6f} Å로 각각 환산.
εxx: a√2={ref['dft']['eps_xx']['mn_a_sqrt2']:.4f}%, c={ref['dft']['eps_xx']['mn_c']:.4f}%.
εyy: a√2={ref['dft']['eps_yy']['mn_a_sqrt2']:.4f}%, c={ref['dft']['eps_yy']['mn_c']:.4f}%.
reference_lines.json에 모든 수치가 있음. 이 값은 Mn 자체 벌크 기준 탄성strain이 아니라
Co 기준 격자 차이를 보여줌. CONTCAR_Co#O4는 코어 기준으로 대체하지 않음:
논문처럼 해당 영상 내부 Co를 0-strain 기준으로 측정하기 때문임.
literature 버전은 논문 Mn a=5.765 Å, c=9.442 Å를 동일한 새 영상 Co 기준으로 환산.
따라서 논문의 0.85%,16.80%와 숫자가 다름. 논문 숫자를 그대로 덧그리지 않음.

5. 아직 원문과 동일하다고 확정할 수 없는 항목과 민감도
미분 폭은 독립 2D primitive valley 벡터의 1.5배, 물리적 폭
{method['derivative']['sigma_physical_nm']:.6f} nm로 둠. 경로상의 가로/세로 1칸과는 다름.
원문에 구체적 격자벡터가 없으므로 가로/세로 간격의 1.5배로 넓힌 경우도 점검.
εxx 차이: 중앙값 {sensitivity[0]['median_abs_difference_pp']:.3f} %p, 최대 {sensitivity[0]['max_abs_difference_pp']:.3f} %p.
εyy 차이: 중앙값 {sensitivity[1]['median_abs_difference_pp']:.3f} %p, 최대 {sensitivity[1]['max_abs_difference_pp']:.3f} %p.
이는 방법 선택 민감도이며 오차막대/신뢰구간이 아님.
Gaussian 영상에는 실제 HAADF의 다중산란, channeling, detector 응답이 없음.
코어 투영에는 앞뒤 Mn shell 신호가 포함됨. 실제 실험 경로와 동일 위치가 아님.

6. 검수
모든 profile 행의 cell_id/label/위치를 원본 피팅 배열과 대조.
εxx, εyy, shear, rotation은 동일한 G에서 추출. 면적의 두 기저 표현 일치 검증.
이동/정상변형/전단/회전의 해석적 미분 점검 최대오차 약1e-14.
최종 그림의 축, 범례, 기준선 라벨 및 이미지 비율은 별도 렌더링 검사.
재실행: scripts/rebuild_fig4_profiles.py → scripts/plot_fig4_revision.py.
'''
    (OUT/'읽어주세요_논문대조와_계산정의.txt').write_text(notes,encoding='utf-8-sig')
    print(json.dumps(dict(reference_n=int(core.sum()),anchor=anchor.tolist(),profile_valid=len(rows),
                         dft_reference_lines=ref['dft'],kernel_sensitivity=sensitivity),indent=2))


if __name__=='__main__':main()
