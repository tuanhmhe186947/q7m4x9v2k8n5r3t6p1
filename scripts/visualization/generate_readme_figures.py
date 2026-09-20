"""Generate clean, minimalist, publication-grade README visual assets.

Design standards:
- Solid white background (#ffffff) with subtle border to ensure 100% visibility on both
  GitHub Light and Dark modes.
- Strictly NO icons, emojis, or decorative clutter.
- Minimalist, modern layout with generous whitespace and clear typography.
- Data-driven directly from authoritative paper tables and evidence ledgers.
- High legibility on mobile and desktop displays.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).parents[2]
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
PAPER_DIR = DOCS_DIR / "paper"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["svg.fonttype"] = "path"


def draw_card_bg(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """Draw a clean solid background card."""
    bg = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=6",
        facecolor="#ffffff",
        edgecolor="#e2e8f0",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(bg)


def generate_pipeline_hero(output_path: Path) -> None:
    """Generate a clean, minimalist 4-stage architecture pipeline diagram."""
    fig, ax = plt.subplots(figsize=(11.0, 2.4), dpi=150, facecolor="#ffffff")
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 180)
    ax.axis("off")

    draw_card_bg(ax, 2, 2, 1096, 176)

    stages = [
        {
            "x": 25,
            "y": 30,
            "w": 195,
            "h": 120,
            "title": "Input Video",
            "line1": "Continuous Farm Feed",
            "line2": "30 FPS Top-Down View",
            "accent": "#475569",
            "bg": "#f8fafc",
        },
        {
            "x": 265,
            "y": 20,
            "w": 260,
            "h": 140,
            "title": "Detection & Tracking",
            "line1": "YOLOv8 Detection",
            "line2": "Online: RealTime-Fast",
            "line3": "Offline: Hybrid-ByteTrack",
            "accent": "#2563eb",
            "bg": "#eff6ff",
            "is_tracking": True,
        },
        {
            "x": 570,
            "y": 30,
            "w": 230,
            "h": 120,
            "title": "Behavior Recognition",
            "line1": "Spatio-Temporal Model",
            "line2": "Spatial-Gated Attention",
            "accent": "#7c3aed",
            "bg": "#f5f3ff",
        },
        {
            "x": 845,
            "y": 30,
            "w": 230,
            "h": 120,
            "title": "Longitudinal Analysis",
            "line1": "10 Behavior Time Budgets",
            "line2": "Near-Zero Profile Distortion",
            "accent": "#059669",
            "bg": "#ecfdf5",
        },
    ]

    for s in stages:
        box = FancyBboxPatch(
            (s["x"], s["y"]),
            s["w"],
            s["h"],
            boxstyle="round,pad=0,rounding_size=8",
            facecolor=s["bg"],
            edgecolor=s["accent"],
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(box)

        cx = s["x"] + s["w"] / 2
        top_y = s["y"] + s["h"] - 26

        ax.text(
            cx,
            top_y,
            s["title"],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=s["accent"],
            zorder=3,
        )

        if s.get("is_tracking"):
            ax.text(
                cx,
                s["y"] + 80,
                s["line1"],
                ha="center",
                va="center",
                fontsize=11,
                color="#1e293b",
                zorder=3,
            )
            ax.text(
                cx,
                s["y"] + 54,
                s["line2"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#1d4ed8",
                zorder=3,
            )
            ax.text(
                cx,
                s["y"] + 30,
                s["line3"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#047857",
                zorder=3,
            )
        else:
            ax.text(
                cx,
                s["y"] + 58,
                s["line1"],
                ha="center",
                va="center",
                fontsize=11,
                color="#334155",
                zorder=3,
            )
            ax.text(
                cx,
                s["y"] + 36,
                s["line2"],
                ha="center",
                va="center",
                fontsize=11,
                color="#64748b",
                zorder=3,
            )

    arrows = [
        (220, 90, 265, 90),
        (525, 90, 570, 90),
        (800, 90, 845, 90),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#64748b",
                lw=1.6,
                mutation_scale=14,
            ),
            zorder=4,
        )

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_tracking_summary(output_path: Path) -> None:
    """Generate a clean, minimalist tracking comparison figure."""
    csv_path = PAPER_DIR / "final_tracking_confirmatory_table.csv"
    data = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["method"]] = {
                "hota": float(row["agg_hota_pct"]),
                "idf1": float(row["agg_idf1_pct"]),
                "idsw": int(row["total_idsw"]),
            }

    methods = [
        ("Raw ByteTrack Baseline", data["Raw ByteTrack"]),
        ("Online RealTime-Fast", data["RealTime-Fast"]),
        ("Offline Hybrid-ByteTrack", data["Hybrid-ByteTrack"]),
    ]

    y_pos = [0, 1, 2]
    method_names = [m[0] for m in methods]
    hota_vals = [m[1]["hota"] for m in methods]
    idf1_vals = [m[1]["idf1"] for m in methods]
    idsw_vals = [m[1]["idsw"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 3.2), dpi=150, facecolor="#ffffff", gridspec_kw={"width_ratios": [2.0, 1.2]}
    )

    # Left plot: Accuracy (HOTA & IDF1)
    ax1.set_facecolor("#ffffff")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(method_names, fontsize=12.5, fontweight="bold", color="#1e293b")
    ax1.set_xlim(88.0, 100.0)
    ax1.grid(axis="x", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax1.set_xlabel("Accuracy (%)  [Higher is better]", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_title("Tracking Accuracy (12 Held-Out Videos)", fontsize=13, fontweight="bold", color="#0f172a", pad=10)

    for y in y_pos:
        ax1.axhline(y, color="#f1f5f9", linewidth=2.0, zorder=1)

    ax1.scatter(hota_vals, y_pos, color="#2563eb", s=110, zorder=3, label="HOTA")
    for x, y in zip(hota_vals, y_pos, strict=True):
        ax1.annotate(
            f"{x:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(-2, 10),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#1d4ed8",
        )

    ax1.scatter(idf1_vals, y_pos, color="#7c3aed", s=110, marker="s", zorder=3, label="IDF1")
    for x, y in zip(idf1_vals, y_pos, strict=True):
        ax1.annotate(
            f"{x:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(2, -16),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#6d28d9",
        )

    ax1.legend(loc="lower right", fontsize=10.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#cbd5e1")
    ax1.spines["bottom"].set_color("#cbd5e1")

    # Right plot: ID switches
    ax2.set_facecolor("#ffffff")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 75)
    ax2.grid(axis="x", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax2.set_xlabel("ID Switches  [Lower is better]", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_title("Identity Switches", fontsize=13, fontweight="bold", color="#0f172a", pad=10)

    colors = ["#f87171", "#fbbf24", "#34d399"]
    bars = ax2.barh(y_pos, idsw_vals, height=0.45, color=colors, zorder=3, edgecolor="#475569", linewidth=0.7)

    labels = ["64", "39 (-39.1%)", "8 (-87.5%)"]
    for bar, label in zip(bars, labels, strict=True):
        w = bar.get_width()
        ax2.text(
            w + 2.0,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
        )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#cbd5e1")
    ax2.spines["bottom"].set_color("#cbd5e1")

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_behavior_cv_summary(output_path: Path) -> None:
    """Generate a clean, minimalist 5-fold CV behavior comparison figure."""
    systems = [
        ("Joint-Representation Baseline", 0.670838, 0.034919, False),
        ("Multimodal Spatio-Temporal Baseline", 0.677789, 0.026822, False),
        ("Class-Aware Spatial-Gated Model", 0.684646, 0.029199, False),
        ("Baseline Fixed Ensemble", 0.692609, 0.040212, False),
        ("Final Spatial-Gated Ensemble", 0.693905, 0.037650, True),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=150, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    y_pos = list(range(len(systems)))
    names = [s[0] for s in systems]
    means = [s[1] for s in systems]
    sds = [s[2] for s in systems]
    is_finals = [s[3] for s in systems]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, fontweight="bold", color="#1e293b")
    ax.set_xlim(0.61, 0.75)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.set_xlabel("Mean Macro-F1 (5-Fold CV)  [Higher is better]", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "5-Fold Cross-Validation Performance (Mean ± SD)",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )

    for y, mean, sd, is_final in zip(y_pos, means, sds, is_finals, strict=True):
        if is_final:
            ax.axhspan(y - 0.35, y + 0.35, color="#eff6ff", zorder=1, alpha=0.9)
            pt_color = "#1d4ed8"
            bar_color = "#1e40af"
        else:
            pt_color = "#475569"
            bar_color = "#64748b"

        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="o",
            color=pt_color,
            ecolor=bar_color,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            markersize=8 if is_final else 7,
            zorder=4,
        )

        tag = "  (Best Ensemble)" if is_final else ""
        ax.text(
            mean + sd + 0.004,
            y,
            f"{mean:.4f} ± {sd:.4f}{tag}",
            va="center",
            fontsize=11,
            fontweight="bold" if is_final else "normal",
            color=pt_color if is_final else "#334155",
            zorder=5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_profile_distortion_summary(output_path: Path) -> None:
    """Generate a clean, minimalist profile distortion comparison figure."""
    systems = [
        ("Raw ByteTrack Baseline", 0.0953, "#f87171"),
        ("Online RealTime-Fast", 0.0309, "#60a5fa"),
        ("Offline Hybrid-ByteTrack", 0.0001, "#34d399"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 2.6), dpi=150, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    y_pos = list(range(len(systems)))
    names = [s[0] for s in systems]
    vals = [s[1] for s in systems]
    colors = [s[2] for s in systems]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, fontweight="bold", color="#1e293b")
    ax.set_xlim(0.00, 0.115)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.set_xlabel("Total Variation Distance (L1)  [Lower is better]", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "Downstream Individual Profile Distortion (DEV12 Cohort)",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )

    bars = ax.barh(y_pos, vals, height=0.42, color=colors, zorder=3, edgecolor="#475569", linewidth=0.7)

    labels = ["0.0953  (High distortion)", "0.0309  (-67.6% error reduction)", "0.0001  (-99.9% near-zero distortion)"]
    for bar, val, label, color in zip(bars, vals, labels, colors, strict=True):
        w = bar.get_width()
        if val < 0.001:
            ax.scatter([val], [bar.get_y() + bar.get_height() / 2], color=color, s=80, zorder=4)
        ax.text(
            w + 0.002,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main() -> None:
    """Regenerate all 4 minimalist README visual SVG assets."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    generate_pipeline_hero(ASSETS_DIR / "system_pipeline.svg")
    generate_tracking_summary(ASSETS_DIR / "tracking_summary.svg")
    generate_behavior_cv_summary(ASSETS_DIR / "behavior_cv_summary.svg")
    generate_profile_distortion_summary(ASSETS_DIR / "profile_distortion_summary.svg")
    print("All 4 minimalist README figures generated successfully.")


if __name__ == "__main__":
    main()
