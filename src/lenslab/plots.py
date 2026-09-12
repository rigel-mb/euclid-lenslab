"""Small, reproducible figures for the Euclid Q1 catalogue audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

NAVY = '#122B45'
TEAL = '#0E9F8F'
BLUE = '#3568D4'
GOLD = '#DFA64A'
MUTED = '#708296'
PALE = '#DDE4EA'
GRADE_COLORS = {'A': TEAL, 'B': BLUE, 'C': GOLD}
FIELDS = ('North', 'Fornax', 'South')

def style():
    """Use one restrained, readable visual language throughout the project."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelcolor': NAVY, 'text.color': NAVY,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': PALE, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'legend.frameon': False,
        'figure.dpi': 110, 'savefig.dpi': 190,
    })

def _numbers(frame, column):
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _save(fig, name, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', bbox_inches='tight')
    return fig

def _heading(fig, title, subtitle):
    fig.suptitle(title, x=.065, y=.99, ha='left', fontsize=19, fontweight='bold')
    fig.text(.065, .99 - .45 / fig.get_figheight(), subtitle, ha='left', va='top', fontsize=10, color=MUTED)

def plot_measurements(candidates, reference, output='figures'):
    """Describe the source populations using the same quantities as matching."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    _heading(fig, 'Before matching: compare the source populations.',
             'The regional 10,000-object sample is context; it does not supply the matched references.')
    prepared = []
    for frame, label, color in [(reference, 'Regional unlabelled sample', MUTED),
                                 (candidates, 'Published candidates', TEAL)]:
        flux = _numbers(frame, 'flux_detection_total')
        area = _numbers(frame, 'segmentation_area')
        # Non-positive values stay in the source tables, but are undefined on these axes.
        measurements = [23.9 - 2.5*np.log10(flux.where(flux.gt(0))),
                        np.log10(area.where(area.gt(0)))]
        prepared.append((measurements, label, color))
    labels = ['Detection-total AB magnitude', 'log₁₀ segmentation area (pixels)']
    for index, (ax, label) in enumerate(zip(axes, labels)):
        finite = [measurements[index].dropna() for measurements, _, _ in prepared]
        bins = np.histogram_bin_edges(pd.concat(finite).to_numpy(), bins=30)
        for (measurements, name, color), values in zip(prepared, finite):
            ax.hist(values, bins=bins, density=True, histtype='step', linewidth=2,
                    color=color, label=f'{name} (n={len(values):,})')
        ax.set(xlabel=label, ylabel='Density')
        ax.legend(fontsize=9)
        ax.grid(axis='y', color=PALE, alpha=.5)
    fig.text(.065, .025,
             'Each histogram integrates to 1. Different selections and sky coverage can explain differences; they are not lensing effects.',
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.75, bottom=.19, left=.075, right=.98, wspace=.25)
    return _save(fig, 'eda_measurements', output)

def plot_gallery(manifest, root, output='figures', n_per_grade=2):
    """Inspect actual science arrays, retaining the original seeded example order."""
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

    style()
    available = manifest.loc[manifest.path_vis.notna()].copy()
    available = available.loc[available.path_vis.map(lambda p: (Path(root) / p).is_file())]
    if any((available.grade == g).sum() < n_per_grade for g in 'ABC'):
        raise ValueError('The gallery download is incomplete; rerun python -m lenslab.data.')
    selected = pd.concat([available[available.grade.eq(g)].head(n_per_grade) for g in 'ABC'])
    fig, axes = plt.subplots(n_per_grade, 3, figsize=(11.5, 3.0*n_per_grade + 1.0), squeeze=False)
    _heading(fig, 'Look at the images behind the labels.',
             'Two examples per grade, in their original seeded order. Real Euclid VIS cutouts.')
    for grade_index, grade in enumerate('ABC'):
        group = selected[selected.grade.eq(grade)]
        for ax, (_, row) in zip(axes[:, grade_index], group.iterrows()):
            data, pixel_scale = _fits_image(Path(root) / row.path_vis)
            finite = data[np.isfinite(data)]
            norm = ImageNormalize(finite, interval=PercentileInterval(99.5), stretch=AsinhStretch(.05))
            ax.imshow(data, origin='lower', cmap='gray', norm=norm)
            height, width = data.shape
            ax.plot([width*.08, width*.08 + 2.0/pixel_scale], [height*.1]*2,
                    color='white', linewidth=2.2, solid_capstyle='butt')
            ax.text(width*.08, height*.15, '2″', color='white', fontsize=8)
            ax.set_title(f'Grade {grade} · …{str(row.object_id)[-7:]}',
                         color=GRADE_COLORS[grade], fontsize=11, loc='left', pad=5)
            ax.set_axis_off()
    fig.text(.065, .018,
             'A/B/C describe visual confidence, not confirmed lens status. Each image uses its own display stretch.',
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.06, left=.06, right=.99, hspace=.20, wspace=.10)
    return _save(fig, 'eda_gallery', output)

def _fits_image(path):
    """Read a nonempty science plane and its WCS angular pixel scale."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    with fits.open(path) as hdus:
        science = next(h for h in hdus if h.data is not None and h.data.ndim == 2)
        data = np.array(science.data, dtype=float)
        scale = float(proj_plane_pixel_scales(WCS(science.header).celestial)[0] * 3600)
    finite = data[np.isfinite(data)]
    if not len(finite) or np.ptp(finite) == 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'Empty science image or invalid WCS: {path}')
    return data, scale

def plot_hero(manifest, root, output='figures', object_id='2740328687682808789'):
    """A labelled crop of an already published candidate for the overview."""
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
    style()
    row = manifest.loc[manifest.object_id.astype(str).eq(object_id)].iloc[0]
    data, scale = _fits_image(Path(root) / row.path_vis)
    h, w = data.shape
    image = data[h//4:3*h//4, w//4:3*w//4]
    norm = ImageNormalize(image[np.isfinite(image)], interval=PercentileInterval(99.5), stretch=AsinhStretch(.04))
    fig, ax = plt.subplots(figsize=(5.6, 5.9))
    ax.imshow(image, origin='lower', cmap='gray', norm=norm)
    h, w = image.shape
    ax.plot([w*.09, w*.09 + 2/scale], [h*.1]*2, color='white', linewidth=2.4)
    ax.text(w*.09, h*.15, '2″', color='white', fontsize=10)
    ax.set_axis_off()
    ax.set_title('Euclid Q1 · published grade A candidate', loc='left', fontsize=12, pad=12)
    fig.text(.05, .027, f'VIS · {object_id} · central ≈8″ crop', color=MUTED, fontsize=9)
    fig.subplots_adjust(left=.05, right=.95, bottom=.065, top=.91)
    return _save(fig, 'eda_hero', output)


def load_data(root):
    """Read the candidate, regional-context and image-gallery tables."""
    root = Path(root)
    frames = []
    for name in ('candidates', 'reference'):
        frame = pd.read_csv(root/'data/processed'/f'{name}.csv', dtype={'object_id': 'string', 'tile_id': 'string'})
        frame = frame.rename(columns={'right_ascension': 'ra', 'declination': 'dec'})
        frame['field'] = frame.field.str.title()
        frames.append(frame)
    path = root/'data/processed/gallery.csv'
    gallery = pd.read_csv(path, dtype={'object_id': 'string'}) if path.exists() else None
    return *frames, gallery


def summarize(candidates, reference, manifest=None):
    """Record availability and displayed measurements, without unrelated analyses."""
    summary = {}
    for name, frame in [('candidates', candidates), ('reference', reference)]:
        flux = _numbers(frame, 'flux_detection_total')
        area = _numbers(frame, 'segmentation_area')
        measurements = {'total_ab_magnitude': 23.9-2.5*np.log10(flux.where(flux>0)),
                        'log10_apparent_area': np.log10(area.where(area>0))}
        summary[name] = {'objects':len(frame), 'fields':frame.field.value_counts().to_dict(),
                         'measurements': {}}
        if name == 'candidates':summary[name]['grades'] = frame.grade.value_counts().to_dict()
        for key, values in measurements.items():
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            summary[name]['measurements'][key] = {'available':len(values),
                'unavailable':len(frame)-len(values), 'median':float(values.median()),
                'q25':float(values.quantile(.25)), 'q75':float(values.quantile(.75))}
    if manifest is not None:
        visible = pd.concat([manifest.loc[manifest.grade.eq(g)].head(2) for g in 'ABC'])
        summary['gallery'] = {'available_source_objects':len(manifest), 'displayed_objects':len(visible),
            'displayed_ids':visible.object_id.astype(str).tolist(),
            'selection':'First two entries per grade from the previously seeded source gallery; no visual selection.'}
    return summary
