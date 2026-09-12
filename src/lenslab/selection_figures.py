"""Draw the retained-pair funnel and balance checks from the frozen sample."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .plots import style

INK = '#122b45'
TEAL = '#087e79'
MUTED = '#64778a'
GOLD = '#cf8739'


def run(root):
    """Create two figures; selection and all counts remain unchanged."""
    root = Path(root)
    summary = json.loads((root / 'results/expanded_sampling_summary.json').read_text())
    candidates = summary['candidate_flow']
    references = summary['reference_pool_flow']
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    style()
    fig, ax = plt.subplots(figsize=(12, 8.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    def box(x, y, title, note, width=.43, height=.105, accent=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle='round,pad=0.012',
            facecolor='#e9f4ef' if accent else '#f4f7f8', edgecolor='#dce5e7'))
        ax.text(x + .018, y + height - .024, title, va='center', fontsize=14,
                weight='bold', color=TEAL if accent else INK)
        ax.text(x + .018, y + height - .052, note, va='top', fontsize=10,
                color=MUTED, linespacing=1.4)

    def arrow(x, upper, lower, label=''):
        ax.annotate('', xy=(x, lower), xytext=(x, upper),
                    arrowprops={'arrowstyle': '->', 'color': MUTED, 'lw': 1.4})
        if label:
            ax.text(x + .018, (upper + lower) / 2, label, va='center',
                    fontsize=9.2, color=MUTED, linespacing=1.4)

    ax.text(.02, .976, 'From public data to 1,126 comparable pairs',
            fontsize=23, weight='bold', color=INK)
    ax.text(.02, .937,
            f"Shared feature source: {references['published_representations_total']:,} published Zoobot vectors; exact object-ID joins.",
            fontsize=10.5, color=MUTED)
    ax.text(.02, .893, 'PUBLISHED CANDIDATES', color=TEAL, fontsize=10, weight='bold')
    ax.text(.55, .893, 'LOCAL UNLABELLED REFERENCES', color=TEAL, fontsize=10, weight='bold')

    box(.02, .75, f"{candidates['published_unique_candidates']:,} candidates", 'Unique catalogue IDs; expert grades A, B and C.')
    box(.55, .75, f"{references['archive_unique_objects_with_common_cuts_near_candidates']:,} archive objects",
        'Within 108″ of eligible candidates;\nalready pass the brightness / size / VIS cuts.')
    arrow(.07, .735, .63, f"−{candidates['missing_representation']:,}: no published vector")
    arrow(.60, .735, .63,
          f"−{references['excluded_published_candidate_ids']:,}: published candidate IDs\n"
          f"−{references['excluded_missing_representation']:,}: no vector\n"
          f"−{references['excluded_within_20arcsec_of_any_published_position']:,}: too close to a candidate (<20″)")

    box(.02, .51, f"{candidates['with_published_representation']:,} with a Zoobot vector", 'Same 40 coordinates for all retained objects.')
    box(.55, .51, f"{references['eligible_reference_pool']:,} possible references",
        'Outside the published list;\nnot verified non-lenses.')
    arrow(.07, .495, .39, f"−{candidates['excluded_common_cuts_after_representation']:,}: fails the common cuts")
    arrow(.60, .495, .39, 'Require similar brightness and apparent size\nfor at least one eligible candidate')

    box(.02, .27, f"{candidates['eligible_candidates']:,} eligible candidates", 'VIS detection; area ≥300 pixels; total AB <22.5.')
    box(.55, .27, f"{references['eligible_references_matching_at_least_one_candidate']:,} admissible references",
        '|Δmag| ≤0.3; |Δlog₁₀ area| ≤0.2;\nwithin 108″ of the matched candidate.')
    arrow(.07, .255, .135,
          f"−{candidates['fewer_than_required_options_before_competition']:,}: no admissible reference\n"
          'One match per candidate; no object reused')
    arrow(.60, .255, .135,
          f"{candidates['retained_references']:,} distinct references allocated\n"
          'using brightness, size and sky separation')

    box(.02, .01,
        f"{candidates['retained_candidates']:,} pairs = {candidates['final_objects']:,} objects",
        'One candidate + one unlabelled reference per pair.\n50% candidates by design; this is not the frequency of lenses in Euclid.',
        width=.96, height=.105, accent=True)
    fig.savefig(out / 'cohort_flow.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    frame = pd.read_csv(root / 'data/processed/expanded_sample.csv', dtype={'object_id': str})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, variable, label in zip(
            axes, ['mag_total', 'log_area', 'snr_total'],
            ['Detection-total AB magnitude', 'log₁₀ segmentation area (pixels)', 'Total-flux signal / noise']):
        for role, color in [('candidate', TEAL), ('reference', GOLD)]:
            values = (frame.loc[frame.role.eq(role), variable]
                      .replace([np.inf, -np.inf], np.nan).dropna().sort_values())
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), lw=2,
                    color=color, label='Candidates' if role == 'candidate' else 'Unlabelled references')
        ax.set(xlabel=label, ylabel='Cumulative fraction' if variable == 'mag_total' else '', ylim=(0, 1))
        if variable == 'snr_total':
            ax.set_xscale('log')
        smd = summary['balance'][variable]['standardized_mean_difference']
        ax.set_title(f'Mean difference / pooled SD: {smd:+.2f}', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('After matching: similar brightness and apparent size',
                 fontsize=18, weight='bold', x=.065, ha='left')
    fig.text(.065, .008,
             '1,126 objects per curve. Signal-to-noise was not used to choose the pairs; observational matching cannot control every difference.',
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / 'expanded_balance.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return summary
