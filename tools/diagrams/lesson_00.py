"""Diagrams for lesson 00 - Start here."""
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("00-start-here")


def features_and_labels():
    fig = S.figure(9, 4.8)
    S.header(fig, "Data is a table: features in, a label out",
             "Each row is one example. The model learns to get from the features (X) to the label (y).")
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.8])
    S.clean_axes(ax)
    cols = ["area (m²)", "rooms", "age (years)", "price ($1,000s)"]
    rows = [[50, 1, 30, 540], [85, 2, 12, 790], [100, 3, 8, 1040], [120, 3, 20, 1080], [140, 4, 5, 1340]]
    x0, cw, gap = 0.2, 0.15, 0.012
    top, rh = 0.62, 0.1
    xs = [x0 + i * (cw + gap) + (0.035 if i == 3 else 0) for i in range(4)]

    # brackets naming the two parts of the table
    for (a, b), color, text in [((0, 2), S.BLUE, "features  (X)  ·  what the model sees"),
                                ((3, 3), S.ORANGE, "label  (y)  ·  the answer")]:
        left, right = xs[a], xs[b] + cw
        ax.plot([left, right], [0.86, 0.86], color=color, lw=3, solid_capstyle="round", transform=ax.transAxes)
        ax.text((left + right) / 2, 0.9, text, ha="center", va="bottom", color=S.INK_2, fontsize=10.5,
                transform=ax.transAxes)

    for i, (name, x) in enumerate(zip(cols, xs)):
        color = S.ORANGE if i == 3 else S.BLUE
        ax.add_patch(FancyBboxPatch((x, top + 0.08), cw, rh, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=color, alpha=0.22, ec="none", transform=ax.transAxes))
        ax.text(x + cw / 2, top + 0.08 + rh / 2, name, ha="center", va="center", color=S.INK,
                fontsize=10.5, fontweight="bold", transform=ax.transAxes)
        for r, row in enumerate(rows):
            y = top - r * (rh + gap)
            ax.add_patch(FancyBboxPatch((x, y - rh + 0.07), cw, rh, boxstyle="round,pad=0,rounding_size=0.015",
                                        fc=S.PANEL, ec="none", transform=ax.transAxes))
            ax.text(x + cw / 2, y - rh / 2 + 0.07, f"{row[i]:,}", ha="center", va="center", color=S.INK_2,
                    fontsize=11, transform=ax.transAxes)

    # call out one row = one example
    r = 2
    y = top - r * (rh + gap) - rh + 0.07
    ax.add_patch(FancyBboxPatch((xs[0] - 0.012, y - 0.008), xs[3] + cw - xs[0] + 0.024, rh + 0.016,
                                boxstyle="round,pad=0,rounding_size=0.02", fc="none", ec=S.INK_2, lw=1.4,
                                transform=ax.transAxes))
    ax.text(xs[0] - 0.03, y + rh / 2, "one row =\none example", ha="right", va="center", color=S.INK_2,
            fontsize=10.5, transform=ax.transAxes)
    S.save(fig, OUT / "features-and-labels.png")


def workflow():
    fig = S.figure(9.6, 5.2)
    S.header(fig, "The six-step recipe behind almost every ML project",
             "You'll repeat these steps in every lesson. Only the model in step 4 changes.")
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.8])
    S.clean_axes(ax)
    steps = [("Collect data", "rows of examples"),
             ("Choose X and y", "the inputs and the answer"),
             ("Split", "train 80%  ·  test 20%"),
             ("Fit", "the model learns from train"),
             ("Predict", "it guesses answers for test"),
             ("Evaluate", "compare guesses with truth")]
    w, h = 0.27, 0.3
    pos = [(0.03, 0.56), (0.365, 0.56), (0.70, 0.56), (0.70, 0.1), (0.365, 0.1), (0.03, 0.1)]
    for k, ((title, cap), (x, y)) in enumerate(zip(steps, pos), start=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=S.PANEL, ec=S.AXIS, lw=1, transform=ax.transAxes))
        ax.scatter([x + 0.045], [y + h - 0.075], s=460, color=S.BLUE if k != 4 else S.ORANGE,
                   transform=ax.transAxes, zorder=3)
        ax.text(x + 0.045, y + h - 0.075, str(k), ha="center", va="center", color=S.INK, fontsize=12,
                fontweight="bold", transform=ax.transAxes, zorder=4)
        ax.text(x + 0.085, y + h - 0.075, title, ha="left", va="center", color=S.INK, fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.03, y + 0.085, cap, ha="left", va="center", color=S.INK_2, fontsize=10.5,
                transform=ax.transAxes)
    for a, b in [((0.30, 0.71), (0.365, 0.71)), ((0.635, 0.71), (0.70, 0.71)),
                 ((0.835, 0.56), (0.835, 0.40)), ((0.70, 0.25), (0.635, 0.25)), ((0.365, 0.25), (0.30, 0.25))]:
        S.arrow(ax, a, b, color=S.INK_2, lw=1.8)
    S.save(fig, OUT / "ml-workflow.png")


def train_test_split():
    fig = S.figure(9, 4.2)
    S.header(fig, "Keep some data hidden: the train / test split",
             "The model studies the training rows. The test rows are the final exam it has never seen.")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.66])
    S.clean_axes(ax)
    n, n_test = 20, 4
    w = 0.9 / n
    for i in range(n):
        test = i >= n - n_test
        ax.add_patch(FancyBboxPatch((0.05 + i * w, 0.46), w - 0.006, 0.3,
                                    boxstyle="round,pad=0,rounding_size=0.008",
                                    fc=S.ORANGE if test else S.BLUE, alpha=0.9, ec="none", transform=ax.transAxes))
    ax.text(0.05 + 8 * w, 0.29, "Training set  ·  80%", ha="center", color=S.INK, fontsize=13,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.05 + 8 * w, 0.14, "the model learns from these rows", ha="center", color=S.INK_2, fontsize=11,
            transform=ax.transAxes)
    ax.text(0.05 + 18 * w, 0.29, "Test set  ·  20%", ha="center", color=S.INK, fontsize=13,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.05 + 18 * w, 0.14, "locked away until the very end", ha="center", color=S.INK_2, fontsize=11,
            transform=ax.transAxes)
    ax.plot([0.05, 0.05 + 16 * w - 0.006], [0.84, 0.84], color=S.BLUE, lw=2, transform=ax.transAxes)
    ax.plot([0.05 + 16 * w, 0.95 - 0.006], [0.84, 0.84], color=S.ORANGE, lw=2, transform=ax.transAxes)
    ax.text(0.5, 0.9, "every row is one example, shuffled before splitting", ha="center", color=S.MUTED,
            fontsize=10, transform=ax.transAxes)
    S.save(fig, OUT / "train-test-split.png")


if __name__ == "__main__":
    features_and_labels()
    workflow()
    train_test_split()
    print("lesson 00 diagrams ->", OUT)
