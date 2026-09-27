"""Course-level images: the README banner (also the social preview), the
roadmap, and the "which algorithm?" cheat sheet."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.ROOT / "assets"

LESSONS = [("00", "Start here"), ("01", "What is ML?"), ("02", "Linear\nregression"),
           ("03", "Logistic\nregression"), ("04", "Trees &\nforests"), ("05", "KNN & SVM"),
           ("06", "Naive Bayes"), ("07", "Clustering\n& PCA"), ("08", "Overfitting\n& CV"),
           ("09", "Final\nproject")]


def count_visuals():
    return sum(1 for p in (S.ROOT / "lessons").glob("*/images/*") if p.suffix in (".png", ".gif"))


def _card(fig, rect):
    x, y, w, h = rect
    # zorder -1: above the background axes (-2) but below the plot axes (0)
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                      transform=fig.transFigure, fc=S.PANEL, ec=S.AXIS, lw=1, zorder=-1))
    ax = fig.add_axes([x + 0.02, y + 0.03, w - 0.04, h - 0.075])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    S.clean_axes(ax)
    ax.patch.set_alpha(0)
    return ax


def banner():
    S.setup()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1])
    grad = np.linspace(0, 1, 256)[:, None]
    bg.imshow(grad, aspect="auto", cmap=S.mpl.colors.LinearSegmentedColormap.from_list(
        "bg", ["#0b1120", "#141d31"]), extent=(0, 1, 0, 1))
    yy, xx = np.mgrid[0:1:200j, 0:1:400j]
    glow = np.exp(-((xx - 0.76) ** 2 / 0.05 + (yy - 0.5) ** 2 / 0.12))
    bg.imshow(glow, extent=(0, 1, 0, 1), cmap=S.mpl.colors.LinearSegmentedColormap.from_list(
        "glow", [to_rgba("#2a78d6", 0), to_rgba("#2a78d6", 0.22)]), aspect="auto")
    S.clean_axes(bg)
    bg.set_zorder(-2)

    fig.text(0.06, 0.74, "Machine Learning,", fontsize=44, fontweight="bold", color=S.INK, va="center")
    fig.text(0.06, 0.6, "Visually", fontsize=44, fontweight="bold", color=S.BRAND, va="center")
    fig.text(0.062, 0.465, "Learn ML from zero. Every idea, drawn.", fontsize=19, color=S.INK_2, va="center")
    x, pad = 0.062, 0.016
    for label in ["10 lessons", f"{count_visuals()} diagrams & animations", "runnable notebooks"]:
        t = fig.text(x + pad, 0.352, label, fontsize=12.5, color=S.INK_2, va="center", fontweight="bold")
        fig.canvas.draw()
        w = t.get_window_extent().width / fig.bbox.width + 2 * pad   # measured, so pills fit their text
        fig.patches.append(FancyBboxPatch((x, 0.315), w, 0.075, boxstyle="round,pad=0,rounding_size=0.035",
                                          transform=fig.transFigure, fc=S.PANEL, ec="#46597a", lw=1.2,
                                          zorder=-1))
        x += w + 0.012
    fig.text(0.064, 0.16, "Python  ·  scikit-learn  ·  free and open source", fontsize=12, color=S.MUTED,
             va="center")

    rng = np.random.default_rng(4)
    # regression
    ax = _card(fig, (0.56, 0.52, 0.2, 0.36))
    xs = rng.uniform(1, 9, 12)
    ax.plot([0.5, 9.5], [1.4, 8.6], color=S.ORANGE, lw=2.2)
    S.scatter(ax, xs, 0.8 * xs + 1 + rng.normal(0, 0.7, 12), 0, s=38)
    ax.set_title("regression", fontsize=10.5, color=S.INK_2, pad=4, fontweight="normal")
    # classification
    ax = _card(fig, (0.775, 0.52, 0.2, 0.36))
    a, b = rng.normal([3, 3], 1, (10, 2)), rng.normal([7, 7], 1, (10, 2))
    xs = np.array([0, 10])
    ax.fill_between(xs, 10 - xs, 0, color=S.BLUE, alpha=0.16, lw=0)
    ax.fill_between(xs, 10 - xs, 10, color=S.ORANGE, alpha=0.16, lw=0)
    S.scatter(ax, a[:, 0], a[:, 1], 0, s=38)
    S.scatter(ax, b[:, 0], b[:, 1], 1, s=38)
    ax.set_title("classification", fontsize=10.5, color=S.INK_2, pad=4, fontweight="normal")
    # clustering
    ax = _card(fig, (0.56, 0.12, 0.2, 0.36))
    for k, c in enumerate([(2.6, 7), (7.3, 6.8), (5, 2.7)]):
        p = rng.normal(c, 0.75, (9, 2))
        S.scatter(ax, p[:, 0], p[:, 1], k, s=38)
    ax.set_title("clustering", fontsize=10.5, color=S.INK_2, pad=4, fontweight="normal")
    # learning curve
    ax = _card(fig, (0.775, 0.12, 0.2, 0.36))
    t = np.linspace(0.3, 9.5, 100)
    ax.plot(t, 1.2 + 7.2 * np.exp(-t / 1.6), color=S.ORANGE, lw=2.2)
    ax.scatter([9.5], [1.2 + 7.2 * np.exp(-9.5 / 1.6)], s=40, color=S.ORANGE, edgecolors=S.PANEL, zorder=3)
    ax.set_title("learning", fontsize=10.5, color=S.INK_2, pad=4, fontweight="normal")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "banner.png", dpi=100)
    plt.close(fig)


def roadmap():
    fig = S.figure(11, 5.6)
    S.header(fig, "The roadmap: ten lessons, zero to your first real project",
             "Each stop has plain-English explanations, diagrams, runnable code, exercises and a quiz.")
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.76])
    S.clean_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * (0.76 * 5.6) / (0.96 * 11))   # equal units on both axes, so circles stay round
    ax.set_aspect("equal")
    phase = {"00": ("Foundations", S.INK_2), "01": ("Foundations", S.INK_2)}
    phase.update({k: ("Supervised learning", S.BLUE) for k in ["02", "03", "04", "05", "06"]})
    phase["07"] = ("Unsupervised learning", S.AQUA)
    phase.update({k: ("Doing it right", S.ORANGE) for k in ["08", "09"]})

    top, bottom = 2.72, 0.98
    xs = [0.9 + i * 1.9 for i in range(5)]
    pos = [(x, top) for x in xs] + [(x, bottom) for x in reversed(xs)]
    track = dict(color=S.AXIS, lw=5, solid_capstyle="round", zorder=1)
    ax.plot([xs[0], xs[-1]], [top, top], **track)
    ax.plot([xs[0], xs[-1]], [bottom, bottom], **track)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
    r = (top - bottom) / 2
    ax.plot(xs[-1] + r * np.cos(theta), bottom + r + r * np.sin(theta), **track)

    for (num, title), (x, y) in zip(LESSONS, pos):
        name, color = phase[num]
        ax.add_patch(Circle((x, y), 0.3, fc=S.SURFACE, ec=color, lw=3, zorder=3))
        ax.text(x, y, num, ha="center", va="center", color=S.INK, fontsize=14, fontweight="bold", zorder=4)
        ax.text(x, y - 0.4, title, ha="center", va="top", color=S.INK_2, fontsize=10.5, linespacing=1.15,
                zorder=4)

    legend = [("Foundations", S.INK_2), ("Supervised learning", S.BLUE), ("Unsupervised learning", S.AQUA),
              ("Doing it right", S.ORANGE)]
    lx = 0.45
    for name, color in legend:
        ax.add_patch(Circle((lx, 3.72), 0.09, fc=S.SURFACE, ec=color, lw=2.5))
        ax.text(lx + 0.18, 3.72, name, va="center", color=S.INK_2, fontsize=10.5)
        lx += 0.45 + 0.1 * len(name)
    S.save(fig, OUT / "roadmap.png")


def cheatsheet():
    fig = S.figure(11, 6.4)
    S.header(fig, "Which algorithm should I use?",
             "A starting point, not a law. Try two or three and compare them with cross-validation (lesson 08).")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.78])
    S.clean_axes(ax)
    cols = [
        ("Predict a number", "regression", S.BLUE, [
            ("Linear regression", "fast, easy to explain; start here", "02"),
            ("Random forest", "when the pattern isn't a straight line", "04"),
            ("KNN", "similar houses, similar prices", "05")]),
        ("Predict a category", "classification", S.ORANGE, [
            ("Logistic regression", "the first thing to try; gives probabilities", "03"),
            ("Random forest", "tables with complex patterns", "04"),
            ("SVM / KNN", "few features, scaled data", "05"),
            ("Naive Bayes", "text, like spam filters", "06")]),
        ("No labels at all", "unsupervised", S.AQUA, [
            ("K-Means", "find groups of similar rows", "07"),
            ("PCA", "shrink many features to a few", "07")]),
    ]
    w, gap, top = 0.315, 0.0225, 0.97
    for i, (title, kind, color, rows) in enumerate(cols):
        x = 0.01 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, top - 0.2, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=S.PANEL, ec=S.AXIS, lw=1, transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((x, top - 0.16), w, 0.16, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=to_rgba(color, 0.22), ec=color, lw=1.4, transform=ax.transAxes))
        ax.text(x + 0.02, top - 0.065, title, color=S.INK, fontsize=13.5, fontweight="bold",
                transform=ax.transAxes, va="center")
        ax.text(x + 0.02, top - 0.122, kind, color=S.INK_2, fontsize=10.5, transform=ax.transAxes,
                va="center")
        for j, (algo, note, lesson) in enumerate(rows):
            y = top - 0.24 - j * 0.14
            ax.text(x + 0.02, y, algo, color=S.INK, fontsize=12, fontweight="bold", transform=ax.transAxes,
                    va="center")
            ax.text(x + w - 0.02, y, f"lesson {lesson}", color=S.MUTED, fontsize=9.5, transform=ax.transAxes,
                    va="center", ha="right")
            ax.text(x + 0.02, y - 0.05, note, color=S.INK_2, fontsize=10, transform=ax.transAxes,
                    va="center")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.13, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=S.SURFACE, ec=S.INK_2, lw=1.2, transform=ax.transAxes))
    ax.text(0.03, 0.085, "Always:", color=S.INK, fontsize=12, fontweight="bold", transform=ax.transAxes,
            va="center")
    ax.text(0.115, 0.085, "split before you train  ·  scale features for KNN, SVM and K-Means  ·  "
            "cross-validate before you trust a score", color=S.INK_2, fontsize=11, transform=ax.transAxes,
            va="center")
    S.save(fig, OUT / "cheatsheet.png")


if __name__ == "__main__":
    banner()
    roadmap()
    cheatsheet()
    print("course images ->", OUT, "| visuals counted:", count_visuals())
