"""Image-neighbour comparison for matched Euclid candidate/reference pairs.

No model is trained on candidate labels. The graph uses the same 40 published
Zoobot coordinates for every object; labels are revealed for the audit only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from . import encoder
from .plots import NAVY, TEAL, MUTED, PALE, style

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"neighbours": 10, "permutations": 1000, "permutation_seed": 20260912}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(root=ROOT):
    root = Path(root)
    sample = pd.read_csv(root / "data/processed/expanded_sample.csv", dtype={"object_id": str, "set_id": str})
    required = {"object_id", "set_id", "role", "grade", "field", "mag_total", "log_area"}
    if not required.issubset(sample.columns):
        raise ValueError(f"Missing sample columns: {required - set(sample.columns)}")
    if sample.object_id.duplicated().any() or sample.set_id.isna().any():
        raise ValueError("Object IDs must be unique and matched sets identified")
    sample = sample.sort_values("object_id", kind="stable").reset_index(drop=True)
    sizes = set(sample.groupby("set_id").size())
    if sizes != {2}:
        raise ValueError("The final analysis requires exactly one candidate and one reference per pair")
    for _, block in sample.groupby("set_id", sort=False):
        if block.role.eq("candidate").sum() != 1 or block.role.eq("reference").sum() != len(block) - 1:
            raise ValueError("Every primary set must have exactly 1 candidate and unlabelled references")
    published = encoder.load_published(root)
    if not sample.object_id.isin(published.index).all():
        raise ValueError("A selected object has no compatible published representation")
    x = published.loc[sample.object_id].to_numpy(dtype=np.float64, copy=True)
    if x.shape != (len(sample), 40) or not np.isfinite(x).all():
        raise ValueError("Expected finite published 40-dimensional coordinates")
    return sample, x


def neighbour_graph(x, set_ids, k=10):
    """Exact Euclidean neighbours; self and every same-set object are excluded.

    Batches keep memory small. Stable distance sorting breaks ties by the input
    order (stable object-ID order), without consulting roles or expert grades.
    """
    set_ids = np.asarray(set_ids)
    if len(x) - max(pd.Series(set_ids).value_counts()) < k:
        raise ValueError("Not enough eligible neighbours outside each matched set")
    graph = np.empty((len(x), k), dtype=np.int32)
    for start in range(0, len(x), 256):
        stop = min(start + 256, len(x))
        distances = pairwise_distances(x[start:stop], x, metric="euclidean")
        distances[set_ids[start:stop, None] == set_ids[None, :]] = np.inf
        graph[start:stop] = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return graph


def audit_neighbours(sample, graph, permutations=1000):
    """Conditional one-candidate-per-set permutations of the fixed graph.

    D = mean_set(f_candidate - mean_reference(f)). Fractions and the identity
    of the anchor candidate are both recomputed under every permuted role map.
    Integer totals preserve exact ties: T=sum(count_i*(q*y_i-1)); D=T/(r*m*k).
    This observational reference requires exchangeability within matched sets;
    it is neither a randomised experiment nor a lens-discovery significance.
    """
    y = sample.role.eq("candidate").to_numpy(dtype=np.int8)
    blocks = list(sample.groupby("set_id", sort=True).indices.values())
    sizes = {len(block) for block in blocks}
    if len(sizes) != 1 or any(y[block].sum() != 1 for block in blocks):
        raise ValueError("Audit requires one candidate per equally sized matched set")
    q, m, k = sizes.pop(), len(blocks), graph.shape[1]
    r = q - 1
    counts = y[graph].sum(axis=1)
    fraction = counts / k
    observed_total = int(np.sum(counts * (q * y - 1), dtype=np.int64))
    denominator = r * m * k
    observed = observed_total / denominator
    result = {"sets": m, "objects": len(sample), "references_per_candidate": r,
              "eligible_candidate_fraction": 1 / q,
              "candidate_neighbour_fraction": float(fraction[y == 1].mean()),
              "reference_neighbour_fraction": float(fraction[y == 0].mean()),
              "difference": observed,
              "candidate_neighbour_enrichment": float(fraction[y == 1].mean() * q),
              "reference_neighbour_enrichment": float(fraction[y == 0].mean() * q),
              "integer_numerator": observed_total, "integer_denominator": denominator,
              "fraction": fraction,
              "interpretation": "Association with prior candidate selection, not precision, recall, accuracy, or lens prevalence.",
              "null_assumption": "One candidate uniformly assigned within each fixed matched set; conditional observational exchangeability is not guaranteed by approximate matching or spatial exclusion."}
    if permutations:
        rng = np.random.default_rng(CONFIG["permutation_seed"])
        block_array = np.stack(blocks)
        null_totals = np.empty(permutations, dtype=np.int64)
        # Recompute both neighbour labels and anchor roles, in bounded batches.
        for start in range(0, permutations, 50):
            stop = min(start + 50, permutations)
            assignment = rng.integers(0, q, size=(stop - start, m))
            swapped = np.zeros((stop - start, len(sample)), dtype=np.int8)
            locations = block_array[np.arange(m)[None, :], assignment]
            swapped[np.arange(stop - start)[:, None], locations] = 1
            null_counts = swapped[:, graph].sum(axis=2)
            null_totals[start:stop] = np.sum(null_counts * (q * swapped - 1), axis=1, dtype=np.int64)
        null = null_totals / denominator
        result.update(permutations=permutations,
                      permutation_p_two_sided=float((1 + (np.abs(null_totals) >= abs(observed_total)).sum()) / (permutations + 1)),
                      null_95_percent_interval=np.quantile(null, [.025, .975]).tolist(), null_mean=float(null.mean()), null=null)
    return result



def plot_neighbours(primary, output):
    """Two anchor populations, one candidate-neighbour fraction."""
    style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    values = [primary["candidate_neighbour_fraction"], primary["reference_neighbour_fraction"]]
    ax.bar([0, 1], values, color=[TEAL, "#A8B5C3"], width=.55, zorder=3)
    for i, value in enumerate(values):
        ax.text(i, value + .025, f"{value:.1%}", ha="center", fontsize=23, weight="bold", color=NAVY)
    ax.axhline(.5, color=MUTED, linestyle=":", linewidth=1.5, zorder=2)
    ax.text(1.45, .51, "50% in the matched corpus", ha="right", color=MUTED, fontsize=9)
    ax.set(xticks=[0, 1], xticklabels=["Around candidates", "Around matched references"],
           ylim=(0, 1), xlim=(-.65, 1.6), ylabel="Candidates among the 10 image neighbours")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color=PALE, alpha=.55, zorder=0)
    fig.suptitle("Known candidates have more candidate neighbours", x=.08, y=.98,
                 ha="left", color=NAVY, fontsize=18, weight="bold")
    fig.text(.08, .89, f'{primary["sets"]:,} matched pairs · difference: {100 * primary["difference"]:.1f} percentage points',
             fontsize=12, color=NAVY)
    fig.text(.08, .035, "The query object and its partner are excluded. These shares describe catalogued candidates, not confirmed lenses.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.1, right=.98, bottom=.16, top=.80)
    fig.savefig(Path(output) / "neighbours.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_permutation(primary, output):
    """Show the conditional reference range without presenting a detection score."""
    style()
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    low, high = np.array(primary["null_95_percent_interval"]) * 100
    difference = 100 * primary["difference"]
    ax.plot([low, high], [0, 0], color="#A8B5C3", lw=12, solid_capstyle="round")
    ax.scatter([0, difference], [0, 0], s=[45, 140], color=[MUTED, TEAL], zorder=4)
    ax.axvline(0, color=PALE, lw=1, zorder=0)
    ax.text(0, .24, f"95% of permuted contrasts\n{low:.1f} to {high:.1f} pp", ha="center", color=MUTED, fontsize=10)
    ax.text(difference, .24, f"Observed\n{difference:.1f} pp", ha="center", color=TEAL, fontsize=12, weight="bold")
    ax.set(xlim=(-8, difference+7), ylim=(-.35, .72), yticks=[],
           xlabel="Candidate minus reference neighbour share (percentage points)")
    ax.spines['left'].set_visible(False)
    fig.suptitle("A conditional check on the observed contrast", x=.08, y=.97,
                 ha="left", color=NAVY, fontsize=17, weight="bold")
    fig.text(.08, .065, "1,000 swaps within fixed pairs; neighbour labels and anchor roles are both recomputed.\nThis reference assumes exchangeable labels within pairs, which observational matching does not guarantee.",
             fontsize=9, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=.08, right=.96, bottom=.29, top=.8)
    fig.savefig(Path(output) / "neighbours_permutation.png", dpi=180, facecolor="white")
    plt.close(fig)


def run(root=ROOT):
    """Recompute exact neighbours and the prespecified matched-pair audit."""
    root = Path(root)
    output = root / "results"
    output.mkdir(exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        sample, x = load_inputs(root)
        graph = neighbour_graph(x, sample.set_id.to_numpy(), CONFIG["neighbours"])
        audit = audit_neighbours(sample, graph, CONFIG["permutations"])
    primary = {key: value for key, value in audit.items() if key not in {"fraction", "null"}}
    table = sample.copy()
    table["candidate_neighbour_fraction"] = audit["fraction"]
    table.to_csv(output / "neighbour_objects.csv", index=False)
    ids = sample.object_id.to_numpy()
    rows = [{"object_id": ids[i], "rank": rank + 1, "neighbour_object_id": ids[j],
             "distance_40d": float(np.linalg.norm(x[i] - x[j]))}
            for i, row in enumerate(graph) for rank, j in enumerate(row)]
    pd.DataFrame(rows).to_csv(output / "neighbours.csv", index=False)
    pd.DataFrame({"difference": audit["null"]}).to_csv(output / "neighbour_permutations.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
        "objects": len(sample), "matched_pairs": int(sample.set_id.nunique()), "dimensions": 40,
        "source_sha": sha256(root / "data/raw/zoobot" / encoder.FILENAME),
        "sample_sha": sha256(root / "data/processed/expanded_sample.csv"),
        "candidate_counts": sample.loc[sample.role.eq("candidate"), "grade"].value_counts().sort_index().to_dict(),
        "representation": "The authors' published 40 PCA coordinates, unchanged; exact Euclidean neighbours; no local training, PCA, whitening or rescaling.",
        "primary": primary,
        "scope_limit": "The corpus is deliberately balanced at one candidate per pair, not representative of Q1 lens prevalence. References are outside the published candidate list, not confirmed non-lenses. Matching and same-pair exclusion do not remove other spatial dependencies or prior candidate-selection biases.",
        "source_url": encoder.RECORD_URL,
    }
    (output / "neighbour_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_neighbours(primary, figures)
    plot_permutation(primary, figures)
    print(f'{len(sample):,} objects / {sample.set_id.nunique():,} pairs: '
          f'{primary["candidate_neighbour_fraction"]:.1%} vs {primary["reference_neighbour_fraction"]:.1%} candidate neighbours.')
    return summary


if __name__ == "__main__":
    run()
