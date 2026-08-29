"""
==============================================================================
25_protocol_diagram.py
Diagrama conceptual del PROTOCOLO transferible de monitoreo (respuesta a la
revision de Environmental Monitoring and Assessment): refuerza visualmente que
el retrieval es un componente de un SISTEMA DE MONITOREO ambiental operativo.

Flujo: campanas de campo (IMARPE/ALT/OEFA) -> match-ups armonizados ->
jerarquia de validacion -> modelo + conformal prediction -> mapa + incertidumbre
-> uso operativo (priorizar muestreo, alerta temprana, gestion transfronteriza).

Salida: results/figures/fig_protocol.png
==============================================================================
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600})

BLUE = "#2e86ab"; GREEN = "#4c9f70"; AMBER = "#edae49"; RED = "#d1495b"
GREY = "#4a4a4a"; LBLUE = "#dce9f2"; LGREEN = "#e4f0e4"; LAMBER = "#fbf0d9"; LRED = "#fbe4e4"


def box(ax, x, y, w, h, text, fc, ec=GREY, fs=8.6, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1.2,
                 edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, lw=1.5, color=GREY, zorder=1))


fig, ax = plt.subplots(figsize=(9.2, 6.0)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

ax.text(6, 7.55, "A transferable protocol for satellite-based water-transparency monitoring",
        ha="center", fontsize=12, fontweight="bold")
ax.text(6, 7.15, "(remote-sensing component complementing in-situ campaigns between field surveys)",
        ha="center", fontsize=9, color=GREY, style="italic")

# Row 1: inputs -> match-ups
box(ax, 2.3, 6.0, 3.4, 1.05,
    "In-situ field campaigns\nIMARPE / ALT / OEFA\n(881 measurements, 156 stations)", LBLUE, fs=8.4)
box(ax, 6.0, 6.0, 3.2, 1.05,
    "Harmonized imagery\nSentinel-2 + Landsat 8/9\n(Roy et al. coefficients)", LBLUE, fs=8.4)
box(ax, 9.7, 6.0, 3.0, 1.05,
    "Satellite-field\nmatch-ups\n(1,002; 812 with Secchi)", BLUE, fs=8.6, tc="white", bold=True)
arrow(ax, 4.05, 6.0, 4.35, 6.0)
arrow(ax, 7.65, 6.0, 8.15, 6.0)

# Row 2: validation hierarchy
box(ax, 5.5, 4.35, 8.8, 1.15,
    "Transferable validation hierarchy\n"
    "random K-fold  $\\approx$  station-wise GroupKFold  |  temporal hold-out  |  leave-one-zone-out",
    LAMBER, ec=AMBER, fs=8.4)
arrow(ax, 9.3, 5.45, 7.5, 4.95)

# Row 3: model + uncertainty
box(ax, 3.4, 2.75, 4.4, 1.05,
    "Random Forest retrieval\n+ split-conformal prediction\n(per-pixel 90% intervals)", LGREEN, ec=GREEN, fs=8.4)
box(ax, 8.6, 2.75, 4.4, 1.05,
    "Lake-wide transparency map\n+ calibrated uncertainty\n(near-weekly cadence)", LGREEN, ec=GREEN, fs=8.4)
arrow(ax, 6.0, 3.77, 6.0, 3.35)
arrow(ax, 5.65, 2.75, 6.35, 2.75)

# Row 4: operational use
box(ax, 6.0, 1.05, 11.2, 1.15,
    "Operational use for environmental monitoring & assessment\n"
    "prioritize field sampling where clarity declines or intervals widen  ·  early-warning trophic proxy (Bahía de Puno)  ·  transboundary Peru–Bolivia management",
    LRED, ec=RED, fs=8.4)
arrow(ax, 3.4, 2.2, 4.5, 1.65)
arrow(ax, 8.6, 2.2, 7.5, 1.65)

# retrievability envelope side-note
ax.text(11.1, 4.35, "Honest\nenvelope:\nSecchi yes;\nChl-a / TSS /\ntemp. no", ha="center", va="center",
        fontsize=7.4, bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec=GREY, lw=1.0))

fig.tight_layout(); fig.savefig(FIG / "fig_protocol.png", bbox_inches="tight"); plt.close(fig)
print("[OK] results/figures/fig_protocol.png")
