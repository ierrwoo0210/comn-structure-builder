"""Full-depth individual-atom Gaussian image proxy (not physical HAADF).

Every final XYZ atom is bilinearly deposited at its transverse coordinate.
Coordinates are never grouped into or averaged within atomic columns. The
deposited weights are convolved in the image plane with normalized Gaussians.
This is a deliberately heuristic incoherent image model: no electron wave,
multislice scattering, channeling, depth selectivity, or detector is simulated.
"""
from pathlib import Path
import argparse
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python_packages'))
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from analyze_paper_strain import read_xyz, FINAL


def bilinear_splat(px, py, weights, x_axis, y_axis, spacing):
    """Return image[row=y, col=x], conserving every individual atom's weight."""
    fx = (px - x_axis[0]) / spacing
    fy = (py - y_axis[0]) / spacing
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    dx, dy = fx - ix, fy - iy
    assert np.all((ix >= 0) & (ix + 1 < len(x_axis)))
    assert np.all((iy >= 0) & (iy + 1 < len(y_axis)))
    image = np.zeros((len(y_axis), len(x_axis)), dtype=np.float64)
    for ox, oy, fraction in (
        (0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
        (0, 1, (1-dx)*dy), (1, 1, dx*dy),
    ):
        np.add.at(image, (iy + oy, ix + ox), weights * fraction)
    expected = float(np.sum(weights, dtype=np.float64))
    assert abs(image.sum(dtype=np.float64) / expected - 1) < 1e-12
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=FINAL)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/strain_gaussian_image')
    parser.add_argument('--spacing', type=float, default=0.1, help='Pixel spacing in angstrom.')
    args = parser.parse_args()
    assert args.spacing > 0
    positions, species, region, source = read_xyz(args.input)
    # Beam is original X: image x = original Y, image y = original Z.
    px, py = positions[:, 1], positions[:, 2]
    atomic_numbers = {'Co': 27, 'Mn': 25, 'O': 8}
    assert set(species) <= set(atomic_numbers)
    Z = np.array([atomic_numbers[s] for s in species], dtype=np.float64)
    sigmas = [0.30, 0.45, 0.60, 0.75]
    truncate_sigma = 6.0
    padding = max(5.0, (truncate_sigma + 2.0) * max(sigmas) + 2 * args.spacing)
    lo = np.floor((np.array([px.min(), py.min()]) - padding) / args.spacing) * args.spacing
    hi = np.ceil((np.array([px.max(), py.max()]) + padding) / args.spacing) * args.spacing
    size = np.rint((hi - lo) / args.spacing).astype(int) + 1
    x_axis = lo[0] + np.arange(size[0]) * args.spacing
    y_axis = lo[1] + np.arange(size[1]) * args.spacing
    arrays = {'x_A': x_axis, 'y_A': y_axis}
    results = {}
    for exponent, widths in ((1.7, sigmas), (2.0, [0.45, 0.60])):
        weights = (Z / 27.0) ** exponent
        splatted = bilinear_splat(px, py, weights, x_axis, y_axis, args.spacing)
        expected = float(weights.sum(dtype=np.float64))
        for sigma in widths:
            name = f'image_s{round(sigma * 100):03d}'
            if exponent != 1.7:
                name += f'_p{round(exponent * 100):03d}'
            convolved = gaussian_filter(splatted, sigma=sigma / args.spacing,
                                        mode='constant', cval=0.0, truncate=truncate_sigma)
            sum_double = float(convolved.sum(dtype=np.float64))
            assert abs(sum_double / expected - 1) < 1e-12, 'Unexpected boundary intensity loss'
            saved = convolved.astype(np.float32)
            sum_saved = float(saved.sum(dtype=np.float64))
            edge_sum = float(np.sum(saved[[0, -1], :], dtype=np.float64)
                             + np.sum(saved[1:-1, [0, -1]], dtype=np.float64))
            assert abs(sum_saved / expected - 1) < 1e-7
            assert edge_sum / expected < 1e-12
            assert np.isfinite(saved).all() and saved.min() >= 0
            arrays[name] = saved
            results[name] = {
                'weight_exponent': exponent,
                'sigma_A': sigma,
                'FWHM_A': float(2 * np.sqrt(2 * np.log(2)) * sigma),
                'weights_by_element': {s: float((z / 27.0) ** exponent)
                                       for s, z in atomic_numbers.items()},
                'expected_intensity_sum': expected,
                'splatted_intensity_sum': float(splatted.sum(dtype=np.float64)),
                'convolved_float64_intensity_sum': sum_double,
                'saved_float32_intensity_sum': sum_saved,
                'saved_relative_sum_error': float(sum_saved / expected - 1),
                'outermost_pixel_border_sum': edge_sum,
                'outermost_pixel_border_fraction': float(edge_sum / expected),
                'max_pixel_intensity': float(saved.max()),
            }
            print(f'{name}: {saved.shape}, sigma={sigma:.2f} A, mass error={sum_saved / expected - 1:.3e}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / 'images.npz'
    np.savez_compressed(image_path, **arrays)
    meta = {
        'model': 'Individual-atom Gaussian projection proxy; NOT physical HAADF-STEM or multislice',
        'source': source,
        'atom_count': int(len(positions)),
        'species_counts': {str(s): int(np.sum(species == s)) for s in np.unique(species)},
        'all_atoms_included': True,
        'pre_average_columns': False,
        'beam_original_axis': 'X [100]',
        'image_horizontal_original_axis': 'Y [010]',
        'image_vertical_original_axis': 'Z [001]',
        'array_convention': 'image[row=y, column=x]; x_A/y_A are pixel-center coordinates in original XYZ angstrom; display origin=lower',
        'pixel_spacing_A': args.spacing,
        'shape_rows_columns': [int(size[1]), int(size[0])],
        'image_x_range_A': [float(x_axis[0]), float(x_axis[-1])],
        'image_y_range_A': [float(y_axis[0]), float(y_axis[-1])],
        'atom_transverse_x_range_A': [float(px.min()), float(px.max())],
        'atom_transverse_y_range_A': [float(py.min()), float(py.max())],
        'padding_at_least_A': padding,
        'gaussian_truncate_sigma': truncate_sigma,
        'intensity_units': 'Arbitrary normalized pixel-integrated signal; each atom contributes total (Z/27)^p.',
        'assumptions': [
            'Each atom scatters independently with heuristic weight (atomic_number/27)^p; exponent is a sensitivity parameter, not fitted to this experiment.',
            'Transverse atom locations retained individually; uniform depth contribution, no coordinate averaging or depth Gaussian.',
            'Bilinear pixel deposition conserves total signal and adds a subpixel discretization blur beyond the stated Gaussian.',
            'The normalized Gaussian acts only in the image plane; sigma values are chosen sensitivity cases, not measured microscope resolution.',
            'All Co, Mn, O included; no region mask in image construction.',
            'No channeling, dynamical scattering, thermal motion, electron-optical aberrations, detector-angle integration, scan noise, or scan drift.',
            'Zero boundary condition with padding beyond the truncated kernel; border and mass conservation explicitly checked.',
        ],
        'images': results,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'archive': str(image_path.resolve()),
    }
    (args.output / 'image_generation_metadata.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {image_path} ({image_path.stat().st_size / 1024**2:.1f} MiB)', flush=True)


if __name__ == '__main__':
    main()
