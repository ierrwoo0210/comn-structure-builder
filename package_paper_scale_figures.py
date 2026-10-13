"""Package the seven reviewed display-only figure revisions and range audit."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.tools/python_packages'),str(ROOT/'.tools/plot_packages')]
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=ROOT/'output/strain_gaussian_image'
OUT=BASE/'paper_scale_7figures'
SOURCE=BASE/'fig4_revision'
NAMES=['εxx 지도','εyy 지도','회전 θ 지도','εxx 코너 프로파일','εyy 코너 프로파일',
       '종횡비 코너 프로파일','면적 코너 프로파일']


def main():
    maps=json.loads((OUT/'map_scale_validation.json').read_text(encoding='utf8'))
    lines=json.loads((OUT/'profile_scale_validation.json').read_text(encoding='utf8'))
    assert maps['source_file_unchanged'] and maps['all_saved_arrays_unchanged']
    assert lines['source_files_unchanged']
    for name,want in lines['source_sha256'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==want
    rows=[]
    for i,item in enumerate(maps['figures']):
        assert item['scatter_values_exactly_equal_saved_values_after_unit_conversion']
        assert item['valid_mask_exactly_equal_saved_mask']
        assert item['xlabel_legend_overlap_check_passed']
        rows.append(dict(그림=NAMES[i],파일=item['output_stem'],단위=item['display_unit'],
            실제최솟값=item['actual_valid_min'],실제최댓값=item['actual_valid_max'],
            표시최솟값=item['display_vmin'],표시최댓값=item['display_vmax'],
            하한미만개수=item['below_count'],상한초과개수=item['above_count'],
            유효점수=item['valid_count'],범위근거='본문 Fig. 4c 색상표의 명시값'))
    for i,item in enumerate(lines['figures'],3):
        assert item['data_unchanged'] and item['checked_line_arrays']==8
        rows.append(dict(그림=NAMES[i],파일=item['stem'],단위=item['unit'] or '무차원',
            실제최솟값=item['actual_min'],실제최댓값=item['actual_max'],
            표시최솟값=item['display_min'],표시최댓값=item['display_max'],
            하한미만개수=item['below'],상한초과개수=item['above'],
            유효점수=item['n'],범위근거='본문 Fig. 4e 눈금과 축 경계에서 판독한 근사 범위'))
    with (OUT/'minmax_and_display_ranges.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    readme='''논문 표시 범위에 맞춘 Gaussian 투영 strain 결과 7장
작성일: 2026-10-12 (Asia/Seoul)

구성
01 εxx 지도 / 02 εyy 지도 / 03 rotation 지도
04 εxx / 05 εyy / 06 종횡비 / 07 unit-cell area 코너 프로파일
각 그림을 PNG와 SVG로 저장했습니다. 프로파일에는 Core와 +1~+7의 8개 경로를 모두 표시했습니다.

이번 변경
계산값, 242점으로 정한 Co-core 기준, 위치, 유효성 마스크는 유지했습니다.
지도 색상 범위와 선형 색상 대응, 선 그래프의 표시 축 범위만 논문과 대조해 바꿨습니다.
각 그림에 실제 계산값의 min/max와 표시 범위 밖의 점 개수를 구분해 적었습니다.

논문에서 확인한 범위
본문 Fig. 4c: εxx/εyy -4~16%, 회전 -2.3~2.3°. SI Fig. S11의 회전 ±1°와 다릅니다.
정상 strain은 -4~16% 선형 매핑이므로 색상표의 중간값은 6%입니다. 0%를 중간색으로 만들지 않았습니다.
색상은 제공된 Fig. 4c 색상표 내부 RGB를 추출하여 재구성했습니다. 원저자 프로그램의 원본 LUT는 아닙니다.
본문 Fig. 4e: x 약 -10~10 unit cells; strain y 약 -2~17%; 종횡비 약 0.86~1.16; 면적 약 0.65~0.77 nm².
Fig. 4e는 축 끝값을 직접 숫자로 명시하지 않아, 눈금과 그림 경계에서 판독한 근사 범위입니다.

범위 밖 값
지도에서는 범위 밖 값을 끝 색으로 표시하고 colorbar 양 끝 삼각형으로 이를 알렸습니다.
선 그래프에서는 범위 밖 점의 x 위치에 빈 삼각형을 표시했습니다. 실제 y 값은 변경하지 않았습니다.
주 축 밖의 데이터 또는 DFT 기준선이 있으면 전체 범위를 보여주는 작은 보조 그래프를 함께 넣었습니다.
특히 면적은 136점 중 88점이 0.77 nm²보다 큽니다. 값이 없어진 것이 아니며 보조 그래프에서 모두 볼 수 있습니다.

기준과 해석
입력은 fig4_revision/revised_data.npz, profiles.csv, reference_lines.json 입니다.
기존 DFT-relaxed Mn bulk / 영상에서 피팅한 Co-core 기준선을 유지했습니다.
논문의 축 범위를 맞추기 위해 DFT 기준선을 논문 수치로 바꾸지 않았습니다.
signed distance의 0은 각 경로의 꺾이는 위치입니다. 1 unit은 같은 행/열의 골 반복 간격 약 0.409 nm입니다.
경로의 실제 자료는 -8~+8까지만 있습니다. -10~10까지 외삽하지 않았습니다.
Gaussian 원자 투영 신호에 기반한 분석이며 물리적 multislice HAADF-STEM 시뮬레이션은 아닙니다.

출처
Oh et al., Design and synthesis of multigrain nanocrystals via geometric misfit strain,
Nature 577, 359–363 (2020), Fig. 4c/e; DOI: 10.1038/s41586-019-1899-3.
제공된 PDF와 tmp/ppt_strain/assets/source의 원문 그림을 대조했습니다.

검수
7개 PNG를 실제로 열어 축/범례/단위/설명 겹침을 검토하고 수정했습니다.
입력 파일 SHA256 불변, 지도에 전달한 수치/위치/유효 마스크 불변,
프로파일 각 8개 곡선의 원본값 일치를 검사했습니다. JSON 파일에 검증 내역이 있습니다.
'''
    (OUT/'읽어주세요_표시범위와_실제값.txt').write_text(readme,encoding='utf-8-sig')

    assets=[]; details=[]
    for row in rows:
        stem=row['파일']
        png=OUT/f'{stem}.png';svg=OUT/f'{stem}.svg'
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            size=list(im.size)
        ET.parse(svg)
        assets.extend([png,svg])
        details.append(dict(stem=stem,png_size=size,png_bytes=png.stat().st_size,
                            svg_bytes=svg.stat().st_size))
    assert len(assets)==14
    final=dict(figure_count=7,png_count=7,svg_count=7,
               source_files_unchanged=True,source_sha256=lines['source_sha256'],
               all_pngs_reopened=True,all_svgs_xml_parsed=True,
               numerical_array_checks_passed=True,visual_review='All seven final figures inspected; overlapping labels repaired.',
               assets=details)
    (OUT/'final_validation.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8')

    # A contact sheet is a preview only; it is not counted as an eighth result.
    fig=plt.figure(figsize=(18,19))
    grid=fig.add_gridspec(3,6,left=.01,right=.99,bottom=.01,top=.99,
                         height_ratios=[1.35,1,1],wspace=.03,hspace=.025)
    positions=[grid[0,0:2],grid[0,2:4],grid[0,4:6],grid[1,0:3],grid[1,3:6],grid[2,0:3],grid[2,3:6]]
    for row,position in zip(rows,positions):
        ax=fig.add_subplot(position)
        with Image.open(OUT/(row['파일']+'.png')) as im:
            ax.imshow(im)
        ax.axis('off')
    preview=BASE/'paper_scale_7figures_preview.png'
    fig.savefig(preview,dpi=180,facecolor='white');plt.close(fig)
    bundle=BASE/'CoMn_strain_paper_scale_7figures.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for asset in assets: z.write(asset,asset.name)
        for name in ['minmax_and_display_ranges.csv','읽어주세요_표시범위와_실제값.txt',
                     'map_scale_validation.json','profile_scale_validation.json','final_validation.json']:
            z.write(OUT/name,name)
        for name in ['profiles.csv','reference_lines.json']:
            z.write(SOURCE/name,'source_values/'+name)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert sum(n.endswith('.png') for n in z.namelist())==7
        assert sum(n.endswith('.svg') for n in z.namelist())==7
    print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,preview=str(preview),figures=7),ensure_ascii=False))


if __name__=='__main__': main()
