"""The shared visual system for every diagram in the course.

Every figure is a dark rounded "card", so it reads the same on GitHub's light
and dark themes. The three categorical colors were checked with a colour-vision
validator against the card surface (all-pairs CVD separation, contrast, and
lightness band all pass), which is why class plots never use a fourth hue.
Classes also get a distinct marker shape, so identity never relies on color.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# Surfaces and ink
SURFACE = "#111827"
PANEL = "#1a2233"        # a raised box drawn on the card
INK = "#ffffff"
INK_2 = "#c8d0dc"
MUTED = "#8b97a8"
GRID = "#1f2937"
AXIS = "#374151"

# Categorical slots (fixed order - never cycled) and their marker shapes
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
SERIES = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "^", "s"]

# Reserved for right / wrong - always shown with a check or cross label
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

# Brand accent for titles on the banner and roadmap only (never for data)
BRAND = "#ffd600"

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#162033", "#104281", "#2a78d6", "#86b6ef", "#cde2fb"])

LINE_W = 1.5      # ~2 px lines at display size
MARK_S = 70       # scatter marker area in pt^2 (~11 px across)
RING_W = 1.4      # surface-coloured ring that keeps overlapping markers legible
WASH = 0.13       # opacity of class-coloured region fills

WATERMARK = "github.com/younespuri/visual-machine-learning-for-beginners"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 10.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "legend.fontsize": 10,
        "lines.linewidth": LINE_W,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "mathtext.fontset": "dejavusans",
    })


def figure(w=9.0, h=5.0):
    setup()
    return plt.figure(figsize=(w, h), dpi=100)


def header(fig, title, subtitle=None):
    fig.text(0.035, 0.955, title, fontsize=16, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.035, 0.885, subtitle, fontsize=11, color=INK_2, va="top")


def scatter(ax, x, y, cls=0, label=None, s=MARK_S, **kw):
    ax.scatter(x, y, s=s, c=SERIES[cls], marker=MARKERS[cls], edgecolors=SURFACE,
               linewidths=RING_W, label=label, zorder=3, **kw)


def clean_axes(ax):
    """Hide ticks and grid for schematic panels where numbers don't matter."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, xy, w, h, text, fc=PANEL, ec=AXIS, color=INK, size=11, weight="normal",
        radius=0.02, lw=1.0, **kw):
    """A rounded box with centred text, in axes-fraction coordinates."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color,
            fontsize=size, fontweight=weight, transform=ax.transAxes, zorder=3, **kw)


def arrow(ax, start, end, color=MUTED, lw=1.6, style="-|>", mutation=14, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=mutation,
                                 color=color, lw=lw, transform=ax.transAxes, zorder=1, **kw))


def save(fig, path, watermark=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if watermark:
        fig.text(0.985, 0.018, WATERMARK, fontsize=7.5, color=MUTED, ha="right", va="bottom")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    round_corners(path)
    return path


def round_corners(path, radius=30):
    im = Image.open(path).convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1),
                                           radius=radius, fill=255)
    im.putalpha(mask)
    im.save(path, optimize=True)


def fig_to_image(fig):
    """Render a figure to a PIL image (used to build animated GIF frames)."""
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def save_gif(frames, path, fps=12, hold_last=24):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames + [frames[-1]] * hold_last
    pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], loop=0,
                      duration=int(1000 / fps), optimize=True, disposal=1)
    return path


def lesson_dir(slug):
    return ROOT / "lessons" / slug / "images"
