"""Diagrams for lesson 01 - What is machine learning?"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("01-what-is-machine-learning")


def ai_ml_dl():
    fig = S.figure(9, 5.2)
    S.header(fig, "AI, ML and DL are nested, not separate",
             "Every deep learning system is machine learning, and all machine learning is AI. Not the other way around.")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.8])
    S.clean_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 3.35)
    ax.set_ylim(-1.08, 1.08)
    rings = [((0, 0), 1.0, S.BLUE, "Artificial Intelligence", 0.78,
              "any machine that acts smart, like a chess\nprogram built from hand-written rules"),
             ((0, -0.3), 0.66, S.ORANGE, "Machine Learning", 0.2,
              "learns its rules from examples,\nlike a spam filter"),
             ((0, -0.6), 0.33, S.AQUA, "Deep Learning", -0.6,
              "learns with many-layered neural\nnetworks, like face recognition")]
    for (cx, cy), r, color, name, ty, note in rings:
        ax.add_patch(Circle((cx, cy), r, fc=to_rgba(color, 0.14), ec=color, lw=1.8))
        ax.text(cx, ty, name, ha="center", va="center", color=S.INK, fontsize=12.5 if r > 0.4 else 11,
                fontweight="bold")
    for (cx, cy), r, color, name, ty, note in rings:
        ax.plot([0.55 if r == 1.0 else r * 0.9, 1.35], [ty, ty], color=S.MUTED, lw=1)
        ax.text(1.42, ty, note, ha="left", va="center", color=S.INK_2, fontsize=10.5, linespacing=1.4)
    S.save(fig, OUT / "ai-ml-dl.png")


def rules_vs_learning():
    fig = S.figure(9.6, 5.0)
    S.header(fig, "Machine learning flips traditional programming around",
             "Same ingredients, swapped places: ML takes the answers as input and gives you the rules.")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.8])
    S.clean_axes(ax)
    rules = dict(fc=to_rgba(S.ORANGE, 0.22), ec=S.ORANGE)
    answers = dict(fc=to_rgba(S.AQUA, 0.22), ec=S.AQUA)
    data = dict(fc=S.PANEL, ec=S.AXIS)
    rows = [(0.6, "Traditional programming", [("Rules", rules), ("Data", data)], "Program", ("Answers", answers),
             "you write the rules"),
            (0.14, "Machine learning", [("Data", data), ("Answers", answers)], "Learning", ("Rules", rules),
             "the computer finds the rules")]
    for y, title, inputs, process, (out_name, out_style), caption in rows:
        ax.text(0.02, y + 0.27, title, color=S.INK, fontsize=13, fontweight="bold", va="center")
        ax.text(0.02 + 0.28, y + 0.27, caption, color=S.MUTED, fontsize=10.5, va="center")
        h = 0.2
        S.box(ax, (0.02, y), 0.14, h, inputs[0][0], size=12, weight="bold", lw=1.4, **inputs[0][1])
        ax.text(0.185, y + h / 2, "+", ha="center", va="center", color=S.INK_2, fontsize=18)
        S.box(ax, (0.21, y), 0.14, h, inputs[1][0], size=12, weight="bold", lw=1.4, **inputs[1][1])
        S.arrow(ax, (0.37, y + h / 2), (0.44, y + h / 2), color=S.INK_2, lw=1.8)
        S.box(ax, (0.45, y), 0.2, h, process, size=12.5, weight="bold", fc=S.SURFACE, ec=S.INK_2, lw=1.4)
        S.arrow(ax, (0.67, y + h / 2), (0.74, y + h / 2), color=S.INK_2, lw=1.8)
        label = out_name if out_name != "Rules" else "Rules\n(a trained model)"
        S.box(ax, (0.75, y), 0.2, h, label, size=12, weight="bold", lw=1.4, **out_style)
    S.save(fig, OUT / "rules-vs-learning.png")


def three_problem_types():
    rng = np.random.default_rng(5)
    fig = S.figure(10, 4.6)
    S.header(fig, "Three kinds of problems you'll solve in this course",
             "Predict a number, predict a category, or find groups nobody labeled.")
    panels = [fig.add_axes([0.045 + i * 0.325, 0.13, 0.27, 0.54]) for i in range(3)]

    ax = panels[0]
    x = rng.uniform(0.5, 9.5, 18)
    y = 0.7 * x + 1.5 + rng.normal(0, 0.8, x.size)
    ax.plot([0, 10], [1.5, 8.5], color=S.ORANGE, label="model")
    S.scatter(ax, x, y, 0, s=45, label="data")
    ax.set_title("Regression")
    ax.set_ylabel("a number  (e.g. price)")

    ax = panels[1]
    a = rng.normal([3, 3], 1.0, (16, 2))
    b = rng.normal([7, 7], 1.0, (16, 2))
    xs = np.linspace(0, 10, 2)
    ax.fill_between(xs, 10 - xs, 0, color=S.BLUE, alpha=S.WASH, lw=0)
    ax.fill_between(xs, 10 - xs, 10, color=S.ORANGE, alpha=S.WASH, lw=0)
    ax.plot(xs, 10 - xs, color=S.INK_2, lw=1.2)
    S.scatter(ax, a[:, 0], a[:, 1], 0, s=45, label="not spam")
    S.scatter(ax, b[:, 0], b[:, 1], 1, s=45, label="spam")
    ax.set_title("Classification")

    ax = panels[2]
    for k, center in enumerate([(2.5, 7.2), (7.4, 7.0), (5.0, 2.6)]):
        pts = rng.normal(center, 0.8, (14, 2))
        S.scatter(ax, pts[:, 0], pts[:, 1], k, s=45, label=f"group {k + 1}")
    ax.set_title("Clustering")

    notes = ["predict a number  ·  supervised", "predict a category  ·  supervised",
             "find groups without labels  ·  unsupervised"]
    for ax, note in zip(panels, notes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
        ax.text(0, 1.02, note, transform=ax.transAxes, color=S.INK_2, fontsize=9.5, va="bottom")
        # legends sit under each panel so they never collide with points or boundaries
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9,
                  handletextpad=0.3, columnspacing=1.2)
    S.save(fig, OUT / "three-problem-types.png")


if __name__ == "__main__":
    ai_ml_dl()
    rules_vs_learning()
    three_problem_types()
    print("lesson 01 diagrams ->", OUT)
