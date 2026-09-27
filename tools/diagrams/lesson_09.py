"""Diagrams for lesson 09 - Final project."""
import sys
from pathlib import Path

from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("09-final-project")

# Where each candidate model was taught in this course
MODEL_LESSON = {"Logistic regression": "lesson 03", "Random forest": "lesson 04",
                "KNN": "lesson 05", "SVM": "lesson 05", "Naive Bayes": "lesson 06"}


def run_project():
    """Re-run the lesson's own code, so every number drawn below is a real result."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                         train_test_split)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, (data.target == 0).astype(int)          # 1 = malignant
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
    }
    pipelines = {name: Pipeline([("scale", StandardScaler()), ("model", model)])
                 for name, model in models.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {name: cross_val_score(p, X_train, y_train, cv=cv) for name, p in pipelines.items()}
    best = max(scores, key=lambda name: scores[name].mean())
    grid = GridSearchCV(pipelines[best], {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=cv)
    grid.fit(X_train, y_train)
    return {"scores": scores, "best": best, "baseline": 1 - y_train.mean(),
            "cm": confusion_matrix(y_test, grid.predict(X_test))}


def luminance(color):
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def ink_on(color):
    """White or dark text, whichever has more contrast on this fill."""
    return max((S.INK, S.SURFACE), key=lambda ink: contrast(ink, color))


def axes_inches(fig, ax):
    pos = ax.get_position()
    return pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()


# ---------------------------------------------------------------- hero
STEPS = [  # (title, detail, where it was taught)
    ("Frame", "what to predict?", "lesson 01"),
    ("Explore", "balance & scales", "lessons 03, 05"),
    ("Split", "80/20, stratified", "lesson 08"),
    ("Scale", "inside a Pipeline", "lesson 05"),
    ("Compare", "5 models, 5 folds", "lessons 03–06, 08"),
    ("Tune", "GridSearchCV", "lesson 08"),
    ("Test once", "114 unseen tumors", "lesson 08"),
    ("Report", "confusion matrix", "lesson 03"),
    ("Explain", "top weights", "lessons 02, 03"),
]
TRAIN_STEPS, TEST_STEP = {3, 4, 5}, 6


def project_pipeline():
    fig = S.figure(10, 5.6)
    S.header(fig, "A machine learning project, end to end",
             "Nine steps, each tagged with the lesson that taught it. "
             "The test set stays locked away until step 7.")
    ax = fig.add_axes([0.035, 0.07, 0.93, 0.72])
    S.clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    w_in, h_in = axes_inches(fig, ax)
    aspect = w_in / h_in                      # keeps rounded corners round in axes units

    inset = 0.004                             # keeps box borders clear of the axes edge
    bw, bh = 0.164, 0.36
    gap = (1 - 2 * inset - 5 * bw) / 4
    col_x = [inset + i * (bw + gap) for i in range(5)]
    top_y, bot_y = 1 - 2 * inset - bh, 2 * inset
    # Top row runs left to right (steps 1-5), bottom row returns right to left (steps 6-9)
    cells = [(col_x[i], top_y) for i in range(5)] + [(col_x[4 - i], bot_y) for i in range(4)]

    for k, ((x, y), (title, detail, tag)) in enumerate(zip(cells, STEPS)):
        edge = S.BLUE if k in TRAIN_STEPS else S.ORANGE if k == TEST_STEP else S.AXIS
        S.box(ax, (x, y), bw, bh, "", ec=edge, lw=1.6 if edge != S.AXIS else 1.0, radius=0.012)
        ax.patches[-1].set_mutation_aspect(aspect)
        cx = x + bw / 2
        ax.text(x + 0.02, y + bh - 0.075, str(k + 1), ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=S.INK, zorder=4,
                bbox=dict(boxstyle="circle,pad=0.28", fc=S.AXIS, ec="none"))
        ax.text(cx, y + bh * 0.60, title, ha="center", va="center", fontsize=14,
                fontweight="bold", color=S.INK, zorder=3)
        ax.text(cx, y + bh * 0.40, detail, ha="center", va="center", fontsize=10.5,
                color=S.INK_2, zorder=3)
        ax.text(cx, y + bh * 0.16, tag, ha="center", va="center", fontsize=9.5, color=S.INK_2,
                zorder=3, bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.75",
                                    fc=S.SURFACE, ec=S.AXIS, lw=0.8))

    pad = 0.008

    def link(a, b, color):
        (xa, ya), (xb, yb) = cells[a], cells[b]
        if ya == yb and xb > xa:                      # rightward along the top row
            S.arrow(ax, (xa + bw + pad, ya + bh / 2), (xb - pad, yb + bh / 2), color=color)
        elif ya == yb:                                # leftward along the bottom row
            S.arrow(ax, (xa - pad, ya + bh / 2), (xb + bw + pad, yb + bh / 2), color=color)
        else:                                         # straight down at the right edge
            S.arrow(ax, (xa + bw / 2, ya - pad), (xb + bw / 2, yb + bh + pad), color=color)

    link(0, 1, S.MUTED)
    link(1, 2, S.MUTED)
    link(2, 3, S.BLUE)
    link(3, 4, S.BLUE)
    link(4, 5, S.BLUE)
    link(5, 6, S.MUTED)
    link(6, 7, S.MUTED)
    link(7, 8, S.MUTED)

    # The test set skips steps 4-6 and only reappears at step 7
    (sx, _), (tx, _) = cells[2], cells[6]
    start, end = (sx + bw * 0.72, top_y - pad), (tx + bw * 0.28, bot_y + bh + pad)
    S.arrow(ax, start, end, color=S.ORANGE, lw=1.8)
    ax.text((start[0] + end[0]) / 2 - 0.03, (start[1] + end[1]) / 2, "test set:\nlocked away",
            ha="right", va="center", fontsize=10.5, color=S.INK_2, linespacing=1.3)

    # Key, in the free corner under step 1
    kx, ky = col_x[0] + 0.005, bot_y + bh - 0.06
    for i, (color, label) in enumerate([(S.BLUE, "training set (80%)"),
                                        (S.ORANGE, "test set (20%)")]):
        yy = ky - i * 0.1
        ax.plot([kx, kx + 0.035], [yy, yy], color=color, lw=2.2, solid_capstyle="round")
        ax.text(kx + 0.05, yy, label, ha="left", va="center", fontsize=10.5, color=S.INK_2)
    yy = ky - 0.215
    ax.text(kx + 0.017, yy, "08", ha="center", va="center", fontsize=9.5, color=S.INK_2,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.75", fc=S.SURFACE, ec=S.AXIS, lw=0.8))
    ax.text(kx + 0.05, yy, "the lesson that\ntaught the step", ha="left", va="center",
            fontsize=10.5, color=S.INK_2, linespacing=1.3)
    S.save(fig, OUT / "project-pipeline.png")


# ---------------------------------------------------------------- model comparison
def rounded_hbar(ax, fig, yc, value, color, thick_in=0.24, radius_in=0.05):
    """A horizontal bar: square at the zero baseline, 4px-style rounded data end."""
    w_in, h_in = axes_inches(fig, ax)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ux, uy = (x1 - x0) / w_in, abs(y1 - y0) / h_in           # data units per inch
    h, rx, ry = thick_in * uy / 2, radius_in * ux, radius_in * uy
    verts = [(0, yc - h), (value - rx, yc - h), (value, yc - h), (value, yc - h + ry),
             (value, yc + h - ry), (value, yc + h), (value - rx, yc + h), (0, yc + h), (0, yc - h)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=3))


def model_comparison(res):
    scores, best = res["scores"], res["best"]
    names = sorted(scores, key=lambda name: -scores[name].mean())   # stable: ties keep code order

    fig = S.figure(9, 5.0)
    S.header(fig, "Cross-validation picks the winner",
             "Mean accuracy over 5 folds of the training set (± the spread between folds). "
             "No test data used.")
    ax = fig.add_axes([0.2, 0.13, 0.76, 0.64])
    ax.set_xlim(0, 1.22)
    ax.set_ylim(len(names) - 0.45, -0.85)                            # winner on top
    ax.set_yticks([])
    ax.grid(False)
    for x in (0.25, 0.5, 0.75, 1.0):                                  # grid stops at the bars
        ax.plot([x, x], [len(names) - 0.45, -0.45], color=S.GRID, lw=0.8, zorder=0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["left"].set_bounds(len(names) - 0.45, -0.45)
    ax.set_xlabel("cross-validation accuracy", x=0.41)

    base = res["baseline"]
    ax.plot([base, base], [len(names) - 0.45, -0.45], color=S.INK_2, lw=1.2, zorder=4,
            solid_capstyle="butt")
    ax.text(base, -0.62, f"baseline {base:.1%}: always say “benign”", ha="center",
            va="bottom", fontsize=10, color=S.INK_2)

    for i, name in enumerate(names):
        mean, std = scores[name].mean(), scores[name].std()
        rounded_hbar(ax, fig, i, mean, S.BLUE)
        ax.text(-0.018, i - 0.08, name, ha="right", va="center", fontsize=11,
                color=S.INK if name == best else S.INK_2)
        ax.text(-0.018, i + 0.24, MODEL_LESSON[name], ha="right", va="center", fontsize=9,
                color=S.MUTED)
        value = ax.text(mean + 0.012, i, f"{mean:.1%}", ha="left", va="center", fontsize=11,
                        color=S.INK, fontweight="bold" if name == best else "normal")
        spread = ax.annotate(f"± {std * 100:.1f}", xy=(1, 0.5), xycoords=value, xytext=(5, 0),
                             textcoords="offset points", ha="left", va="center", fontsize=9.5,
                             color=S.MUTED)
        if name == best:
            ax.annotate("winner", xy=(1, 0.5), xycoords=spread, xytext=(9, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=10,
                        fontweight="bold", color=S.INK,
                        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=S.PANEL,
                                  ec=S.AXIS, lw=0.8))
    S.save(fig, OUT / "model-comparison.png")


# ---------------------------------------------------------------- confusion matrix
def confusion_matrix_card(res):
    cm = res["cm"]
    (tn, fp), (fn, tp) = cm
    total = cm.sum()
    cells = {  # (row = truth, col = prediction): (caption, how to read it, right?)
        (0, 0): ("correctly cleared", "benign, called benign", True),
        (0, 1): ("false alarm", "benign, called malignant", False),
        (1, 0): ("missed cancer", "malignant, called benign", False),
        (1, 1): ("cancer caught", "malignant, called malignant", True),
    }

    fig = S.figure(9, 5.4)
    S.header(fig, f"The final grade: {tn + tp} of {total} right",
             f"The tuned model on the {total} test tumors it had never seen. "
             "Rows are the truth; columns are the model's answer.")
    ax = fig.add_axes([0.2, 0.06, 0.44, 0.64])
    S.clean_axes(ax)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    w_in, h_in = axes_inches(fig, ax)
    g = 0.012                                                   # surface gap between cells

    for (r, c), (caption, reading, right) in cells.items():
        n = cm[r, c]
        fill = S.SEQ(0.14 + 0.86 * n / cm.max())
        ink = ink_on(fill)
        patch = FancyBboxPatch((c + g, r + g), 1 - 2 * g, 1 - 2 * g,
                               boxstyle="round,pad=0,rounding_size=0.035", fc=fill, ec="none",
                               mutation_aspect=(w_in / 2) / (h_in / 2), zorder=2)
        ax.add_patch(patch)
        ax.text(c + 0.5, r + 0.43, str(n), ha="center", va="center", fontsize=32,
                fontweight="bold", color=ink, zorder=3)
        ax.text(c + 0.5, r + 0.68, caption, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ink, zorder=3)
        ax.text(c + 0.5, r + 0.84, reading, ha="center", va="center", fontsize=9.5,
                color=ink, alpha=0.85, zorder=3)
        badge = ax.text(c + 0.1, r + 0.14, "✓" if right else "✗", ha="center",
                        va="center", fontsize=10, fontweight="bold", color=S.INK, zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.25",
                                  fc=S.GOOD if right else S.CRITICAL, ec="none"))
        ax.annotate("right" if right else "wrong", xy=(1, 0.5), xycoords=badge, xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=10,
                    fontweight="bold", color=ink, zorder=4)

    for i, label in enumerate(["benign", "malignant"]):
        ax.text(i + 0.5, -0.05, label, ha="center", va="bottom", fontsize=11, color=S.INK_2)
        ax.text(-0.04, i + 0.5, label, ha="right", va="center", fontsize=11, color=S.INK_2)
    ax.text(1, -0.2, "the model said", ha="center", va="bottom", fontsize=10, color=S.MUTED)
    ax.text(-0.04, -0.2, "the truth was", ha="right", va="bottom", fontsize=10, color=S.MUTED)

    # Three ways to read the same four numbers
    stats = [(f"{(tn + tp) / total:.1%}", "accuracy", f"{tn + tp} of {total} tumors right"),
             (f"{tp / (tp + fn):.1%}", "recall (malignant)", f"caught {tp} of {tp + fn} cancers"),
             (f"{tp / (tp + fp):.1%}", "precision (malignant)",
              f"{tp} of {tp + fp} alarms were real")]
    x0, y0 = 0.675, 0.66
    for k, (value, name, meaning) in enumerate(stats):
        y = y0 - k * 0.175
        fig.text(x0, y, value, fontsize=21, fontweight="bold", color=S.INK, va="center")
        fig.text(x0 + 0.125, y + 0.018, name, fontsize=10.5, color=S.INK, va="center")
        fig.text(x0 + 0.125, y - 0.022, meaning, fontsize=9.5, color=S.INK_2, va="center")
    fig.text(x0, y0 - 3 * 0.175 + 0.02,
             f"For cancer screening, the {fn} misses\nmatter far more than the {fp} false alarm.",
             fontsize=10.5, color=S.INK_2, va="center", linespacing=1.4)
    S.save(fig, OUT / "confusion-matrix.png")


if __name__ == "__main__":
    results = run_project()
    project_pipeline()
    model_comparison(results)
    confusion_matrix_card(results)
    print("lesson 09 diagrams ->", OUT)
