"""Diagrams for lesson 04 - Decision trees and random forests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("04-decision-trees-and-random-forests")

CLASS_NAMES = ["No flu", "Flu"]

# The small tree the lesson's code learns in Step 2 (thresholds copied from its export_text)
T_CUT, F_LEFT, F_RIGHT = 37.75, 8.5, 2.5
T_RANGE, F_RANGE = (36.0, 39.8), (0.5, 10.5)


def wash(cls, boost=0.0):
    return to_rgba(S.SERIES[cls], S.WASH + boost)


def tag(ax, x, y, text, transform=None, size=10):
    """A small rounded label such as 'Q1' that links a question to its cut."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight="bold", color=S.INK,
            transform=transform or ax.transData, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=S.PANEL, ec=S.INK_2, lw=1.0))


def marker(ax, x, y, cls, s=90, transform=None):
    ax.scatter([x], [y], s=s, c=S.SERIES[cls], marker=S.MARKERS[cls], edgecolors=S.SURFACE,
               linewidths=S.RING_W, zorder=5, transform=transform or ax.transData, clip_on=False)


def leaf(ax, cx, cy, w, h, cls, size=11):
    """A leaf box: class wash, class marker, class name in ink."""
    S.box(ax, (cx - w / 2, cy - h / 2), w, h, "", fc=wash(cls, 0.05), ec=S.SERIES[cls], lw=1.2)
    marker(ax, cx - w * 0.26, cy, cls, s=80, transform=ax.transAxes)
    ax.text(cx + w * 0.08, cy, CLASS_NAMES[cls], ha="center", va="center", color=S.INK,
            fontsize=size, fontweight="bold", transform=ax.transAxes, zorder=4)


def hero_points():
    """A clean toy sample laid out so the Step 2 tree carves it into its four boxes."""
    rng = np.random.default_rng(4)
    regions = [  # (temperature range, fatigue range, class, how many)
        ((36.15, 37.6), (0.8, 8.2), 0, 20),
        ((36.15, 37.6), (8.8, 10.2), 1, 5),
        ((37.9, 39.65), (0.8, 2.2), 0, 5),
        ((37.9, 39.65), (2.8, 10.2), 1, 22),
    ]
    rows = []
    for (t0, t1), (f0, f1), cls, k in regions:
        rows += [(t, f, cls) for t, f in zip(rng.uniform(t0, t1, k), rng.uniform(f0, f1, k))]
    # A few patients who don't follow the pattern: no small tree is ever perfect
    rows += [(37.1, 5.6, 1), (38.9, 7.4, 0), (36.7, 2.1, 1)]
    return np.array(rows)


def tree_as_questions():
    fig = S.figure(12, 5.8)
    S.header(fig, "A decision tree is a flowchart of yes/no questions",
             "Each question makes one straight cut through the data. Match Q1, Q2 and Q3 on both sides.")

    # Left: the flowchart
    ax = fig.add_axes([0.02, 0.05, 0.47, 0.74])
    S.clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    qs = [  # (centre x, centre y, width, text, tag)
        (0.50, 0.86, 0.56, "Temperature > 37.75 °C?", "Q1"),
        (0.25, 0.53, 0.40, "Fatigue > 8.5?", "Q2"),
        (0.75, 0.53, 0.40, "Fatigue > 2.5?", "Q3"),
    ]
    qh = 0.13
    for cx, cy, w, text, t in qs:
        S.box(ax, (cx - w / 2, cy - qh / 2), w, qh, text, size=11.5, weight="bold", lw=1.2, ec=S.INK_2)
        tag(ax, cx - w / 2, cy + qh / 2, t, transform=ax.transAxes)
    # root -> children
    for cx, label in [(0.25, "no"), (0.75, "yes")]:
        S.arrow(ax, (0.5 + (cx - 0.5) * 0.35, 0.86 - qh / 2), (cx, 0.53 + qh / 2 + 0.01))
        ax.text(0.5 + (cx - 0.5) * 0.86, 0.745, label, ha="center", va="center", color=S.INK_2,
                fontsize=10.5, style="italic", transform=ax.transAxes)
    # children -> leaves
    leaves = [(0.125, 0), (0.375, 1), (0.625, 0), (0.875, 1)]
    lw_, lh = 0.225, 0.13
    for i, (lx, cls) in enumerate(leaves):
        parent_x = 0.25 if i < 2 else 0.75
        start = (parent_x + (lx - parent_x) * 0.3, 0.53 - qh / 2)
        S.arrow(ax, start, (lx, 0.17 + lh / 2 + 0.01))
        ax.text((start[0] + lx) / 2 + (-0.035 if lx < parent_x else 0.035), 0.375,
                "no" if lx < parent_x else "yes", ha="center", va="center", color=S.INK_2,
                fontsize=10.5, style="italic", transform=ax.transAxes)
        leaf(ax, lx, 0.17, lw_, lh, cls)
    ax.text(0.5, 0.02, "Leaves hold the answers", ha="center", va="bottom", color=S.MUTED,
            fontsize=10, transform=ax.transAxes)

    # Right: the same questions as cuts through the data
    ax2 = fig.add_axes([0.57, 0.12, 0.40, 0.62])
    (t0, t1), (f0, f1) = T_RANGE, F_RANGE
    boxes = [((t0, f0), T_CUT - t0, F_LEFT - f0, 0), ((t0, F_LEFT), T_CUT - t0, f1 - F_LEFT, 1),
             ((T_CUT, f0), t1 - T_CUT, F_RIGHT - f0, 0), ((T_CUT, F_RIGHT), t1 - T_CUT, f1 - F_RIGHT, 1)]
    for (x, y), w, h, cls in boxes:
        ax2.add_patch(Rectangle((x, y), w, h, fc=wash(cls), ec="none", zorder=0))
    ax2.plot([T_CUT, T_CUT], [f0, f1], color=S.INK, lw=2.2, zorder=2)
    ax2.plot([t0, T_CUT], [F_LEFT, F_LEFT], color=S.INK, lw=2.2, zorder=2)
    ax2.plot([T_CUT, t1], [F_RIGHT, F_RIGHT], color=S.INK, lw=2.2, zorder=2)
    pts = hero_points()
    for cls in (0, 1):
        m = pts[:, 2] == cls
        S.scatter(ax2, pts[m, 0], pts[m, 1], cls, label=CLASS_NAMES[cls], s=60)
    tag(ax2, T_CUT, f1 + 0.02, "Q1")
    tag(ax2, t0 + 0.02, F_LEFT, "Q2")
    tag(ax2, t1 - 0.02, F_RIGHT, "Q3")
    ax2.set_xlim(t0, t1)
    ax2.set_ylim(f0, f1)
    ax2.grid(False)
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Fatigue (1 to 10)")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2, handletextpad=0.3,
               columnspacing=1.4, borderaxespad=0)
    S.save(fig, OUT / "tree-as-questions.png")


def gini(n_flu, n_no):
    p = n_flu / (n_flu + n_no)
    return 1 - (p ** 2 + (1 - p) ** 2)


def group_box(ax, cx, cy, w, h, n_flu, n_no, title=None, per_row=5):
    """A group of patients drawn as markers, with its Gini worked out underneath."""
    S.box(ax, (cx - w / 2, cy - h / 2), w, h, "", lw=1.1)
    cls = [1] * n_flu + [0] * n_no
    rows = [cls[i:i + per_row] for i in range(0, len(cls), per_row)]
    top = cy + h / 2 - (0.075 if title else 0.05)
    if title:
        ax.text(cx, cy + h / 2 - 0.03, title, ha="center", va="center", color=S.INK_2, fontsize=10.5)
    for r, row in enumerate(rows):
        xs = cx + (np.arange(len(row)) - (len(row) - 1) / 2) * 0.03
        for x, c in zip(xs, row):
            marker(ax, x, top - r * 0.052, c, s=85)
    p = n_flu / (n_flu + n_no)
    g = gini(n_flu, n_no)
    base = top - (len(rows) - 1) * 0.052
    ax.text(cx, base - 0.058, f"Gini = {g:.2f}" + ("  (pure)" if g == 0 else ""), ha="center",
            va="center", color=S.INK, fontsize=11.5, fontweight="bold")
    ax.text(cx, base - 0.105, f"1 − ({p:.1f}² + {1 - p:.1f}²)", ha="center", va="center",
            color=S.MUTED, fontsize=10)


def gini_split():
    fig = S.figure(12, 6.8)
    S.header(fig, "How a tree picks a question: the purest split wins",
             "It tries every possible question and keeps the one whose groups have the lowest Gini impurity.")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.78])
    S.clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    group_box(ax, 0.5, 0.855, 0.2, 0.25, 4, 6, title="10 patients")
    candidates = [  # (column centre, question, [(answer, n_flu, n_no), ...], verdict)
        (0.25, "Question A:  Temperature > 38 °C?", [("no", 0, 5), ("yes", 4, 1)], True),
        (0.75, "Question B:  Cough?", [("no", 1, 4), ("yes", 3, 2)], False),
    ]
    for cx, question, kids, best in candidates:
        S.arrow(ax, (0.5 + (cx - 0.5) * 0.25, 0.73), (cx, 0.625), linestyle=(0, (4, 3)))
        S.box(ax, (cx - 0.17, 0.555), 0.34, 0.07, question, size=11.5, weight="bold",
              ec=S.INK_2, lw=1.2)
        ginis = []
        for k, (answer, n_flu, n_no) in enumerate(kids):
            kx = cx + (k - 0.5) * 0.22
            S.arrow(ax, (cx + (k - 0.5) * 0.08, 0.555), (kx, 0.455))
            ax.text((cx + (k - 0.5) * 0.08 + kx) / 2 + (k - 0.5) * 0.05, 0.505, answer, ha="center",
                    va="center", color=S.INK_2, fontsize=10.5, style="italic")
            group_box(ax, kx, 0.345, 0.19, 0.2, n_flu, n_no)
            ginis.append(gini(n_flu, n_no))
        avg = np.mean(ginis)
        line = f"Average Gini  ({ginis[0]:.2f} + {ginis[1]:.2f}) / 2 = {avg:.2f}"
        ax.text(cx, 0.165, line, ha="center", va="center", color=S.INK, fontsize=11.5)
        if best:
            ax.text(cx, 0.1, "✓  Lowest: the tree asks this question", ha="center", va="center",
                    color=S.GOOD, fontsize=11.5, fontweight="bold")
        else:
            ax.text(cx, 0.1, "Groups still mixed: not chosen", ha="center", va="center",
                    color=S.MUTED, fontsize=11)
    ax.text(0.5, 0.015, "Gini = 1 − (share with flu² + share without flu²)     0 means pure     "
            "0.5 is a 50/50 mix", ha="center", va="bottom", color=S.MUTED, fontsize=10)
    handles = [Line2D([], [], ls="", marker=S.MARKERS[c], ms=9, mfc=S.SERIES[c], mec=S.SURFACE,
                      label=CLASS_NAMES[c]) for c in (0, 1)]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, borderaxespad=0.2)
    S.save(fig, OUT / "gini-split.png")


def noisy_patients(seed=242, n=240, flip=0.12):
    """Two features, a diagonal true boundary, and 12% of labels flipped (misdiagnosed)."""
    rng = np.random.default_rng(seed)
    temp = np.clip(rng.normal(37.7, 0.75, n), 36.1, 39.7)
    fatigue = rng.uniform(0.6, 10.4, n)
    y = (2 * (temp - 36) + 0.45 * fatigue > 5.8).astype(int)
    wrong = rng.random(n) < flip
    y[wrong] = 1 - y[wrong]
    return np.column_stack([temp, fatigue]), y


def tree_depth():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = noisy_patients()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    (t0, t1), (f0, f1) = (36.0, 39.8), (0.4, 10.6)
    gx, gy = np.meshgrid(np.linspace(t0, t1, 700), np.linspace(f0, f1, 700))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    fig = S.figure(12.5, 5.9)
    S.header(fig, "Deeper trees draw more boxes, until they box in single patients",
             "Three trees trained on the same noisy patients (the dots). With no limit, the tree aces "
             "them but does worse on new patients.")
    panels = [(1, "max_depth = 1", "Too simple: one cut"),
              (4, "max_depth = 4", "About right: follows the trend"),
              (None, "No depth limit", "Memorized: tiny boxes around flukes")]
    for i, (depth, title, verdict) in enumerate(panels):
        model = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
        ax = fig.add_axes([0.055 + i * 0.318, 0.25, 0.27, 0.52])
        pred = model.predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5], colors=[S.BLUE, S.ORANGE], alpha=S.WASH)
        ax.contour(gx, gy, pred, levels=[0.5], colors=[S.INK_2], linewidths=0.9)
        for cls in (0, 1):
            m = y_train == cls
            S.scatter(ax, X_train[m, 0], X_train[m, 1], cls, s=34, label=CLASS_NAMES[cls])
        ax.set_xlim(t0, t1)
        ax.set_ylim(f0, f1)
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel("Temperature (°C)")
        if i == 0:
            ax.set_ylabel("Fatigue (1 to 10)")
        tr, te = model.score(X_train, y_train), model.score(X_test, y_test)
        fig.text(0.055 + i * 0.318, 0.115, f"{verdict}\n{model.get_n_leaves()} leaves   ·   "
                 f"train {tr:.0%}   ·   test {te:.0%}", color=S.INK_2, fontsize=10.5,
                 va="center", linespacing=1.6)
    handles = [Line2D([], [], ls="", marker=S.MARKERS[c], ms=8, mfc=S.SERIES[c], mec=S.SURFACE,
                      label=CLASS_NAMES[c]) for c in (0, 1)]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.962, 0.806), ncol=2,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    S.save(fig, OUT / "tree-depth.png")


def lesson_patients():
    """The 500 made-up patients from the lesson's Step 1, split exactly as in Step 2."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(367)
    n = 500
    df = pd.DataFrame({
        "temperature": rng.normal(37.7, 0.8, n).round(1),
        "fatigue": rng.integers(1, 11, n),
        "cough": rng.integers(0, 2, n),
        "aches": rng.integers(0, 2, n),
        "chills": rng.integers(0, 2, n),
        "sore_throat": rng.integers(0, 2, n),
    })
    points = (2 * (df.temperature - 36) + 1.5 * df.cough + df.aches + df.chills
              + df.sore_throat + df.fatigue / 4)
    df["flu"] = (points > 7).astype(int)
    wrong = rng.random(n) < 0.12
    df.loc[wrong, "flu"] = 1 - df.loc[wrong, "flu"]
    X, y = df.drop(columns="flu"), df["flu"]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# The new patient from the lesson's Step 6
PATIENT = {"temperature": 38.5, "fatigue": 5, "cough": 0, "aches": 1, "chills": 0, "sore_throat": 0}
BINARY = {"cough": "Cough?", "aches": "Aches?", "chills": "Chills?", "sore_throat": "Sore throat?"}


def question(feature, threshold):
    if feature in BINARY:
        return BINARY[feature]
    if feature == "temperature":
        return f"Temp > {threshold:.2f} °C?"
    return f"Fatigue > {threshold:.1f}?"


def random_forest_vote():
    from sklearn.ensemble import RandomForestClassifier

    X_train, _, y_train, _ = lesson_patients()
    cols = list(X_train.columns)
    forest = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=191).fit(X_train, y_train)
    patient = np.array([[PATIENT[c] for c in cols]], dtype=float)

    fig = S.figure(12.5, 7.0)
    S.header(fig, "A random forest: many different trees, one vote",
             "Five small trees, each trained on its own random sample of patients and a random few "
             "features per question. Each one votes; the majority wins.")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.8])
    S.clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    faded = dict(color=S.MUTED, alpha=0.35, lw=1.2)

    S.box(ax, (0.21, 0.885), 0.58, 0.085,
          "New patient:   38.5 °C   ·   fatigue 5   ·   body aches   ·   no cough, chills or sore throat",
          size=11.5, weight="bold", ec=S.INK_2, lw=1.2)
    votes = []
    for k, tree in enumerate(forest.estimators_):
        T = tree.tree_
        cx = 0.1 + k * 0.2
        S.arrow(ax, (0.5 + (cx - 0.5) * 0.5, 0.885), (cx, 0.832), color=S.MUTED, lw=1.0, mutation=10)
        ax.text(cx, 0.808, f"Tree {k + 1}", ha="center", va="center", color=S.INK_2, fontsize=10.5)
        # root question
        S.box(ax, (cx - 0.08, 0.7), 0.16, 0.075, question(cols[T.feature[0]], T.threshold[0]),
              size=10, weight="bold", ec=S.INK_2, lw=1.2)
        go_right = patient[0, T.feature[0]] > T.threshold[0]
        for side, child in enumerate((T.children_left[0], T.children_right[0])):
            on_path = (side == 1) == go_right
            kx = cx + (side - 0.5) * 0.07
            if on_path:
                S.arrow(ax, (cx + (side - 0.5) * 0.06, 0.7), (kx, 0.585), color=S.INK_2, lw=1.8)
                ax.text((cx + (side - 0.5) * 0.06 + kx) / 2 + (side - 0.5) * 0.045, 0.645,
                        "yes" if side else "no", ha="center", va="center", color=S.INK_2,
                        fontsize=9.5, style="italic")
                S.box(ax, (kx - 0.05, 0.51), 0.1, 0.075, question(cols[T.feature[child]], T.threshold[child]),
                      size=9.5, weight="bold", ec=S.INK_2, lw=1.2)
                go_right2 = patient[0, T.feature[child]] > T.threshold[child]
                for side2, leaf_id in enumerate((T.children_left[child], T.children_right[child])):
                    lx = kx + (side2 - 0.5) * 0.06
                    cls = int(np.argmax(T.value[leaf_id]))
                    if (side2 == 1) == go_right2:
                        S.arrow(ax, (kx + (side2 - 0.5) * 0.04, 0.51), (lx, 0.425), color=S.INK_2, lw=1.8)
                        ax.text((kx + (side2 - 0.5) * 0.04 + lx) / 2 + (side2 - 0.5) * 0.035, 0.47,
                                "yes" if side2 else "no", ha="center", va="center", color=S.INK_2,
                                fontsize=9.5, style="italic")
                        marker(ax, lx, 0.39, cls, s=190)
                        votes.append(cls)
                        ax.text(cx, 0.305, f"votes {CLASS_NAMES[cls]}", ha="center", va="center",
                                color=S.INK, fontsize=11, fontweight="bold")
                    else:
                        ax.plot([kx + (side2 - 0.5) * 0.04, lx], [0.51, 0.415], **faded)
                        ax.scatter([lx], [0.39], s=55, c=S.SERIES[cls], marker=S.MARKERS[cls],
                                   alpha=0.3, edgecolors="none", zorder=3)
            else:  # a branch this patient never visits: drawn faint, without its question
                sx = cx + (side - 0.5) * 0.12
                ax.plot([cx + (side - 0.5) * 0.07, sx], [0.7, 0.61], **faded)
                ax.add_patch(Rectangle((sx - 0.016, 0.565), 0.032, 0.045, fc="none", ec=S.MUTED,
                                       alpha=0.35, lw=1.2, zorder=2))
                ax.text(sx, 0.5875, "…", ha="center", va="center", color=S.MUTED, alpha=0.6, fontsize=10)

    # the tally
    S.box(ax, (0.12, 0.07), 0.76, 0.14, "", ec=S.INK_2, lw=1.2)
    ax.text(0.15, 0.14, "Count the votes:", ha="left", va="center", color=S.INK_2, fontsize=11.5)
    for cls, x0 in ((1, 0.3), (0, 0.475)):
        n_votes = votes.count(cls)
        for j in range(n_votes):
            marker(ax, x0 + j * 0.026, 0.14, cls, s=150)
        ax.text(x0 + n_votes * 0.026 - 0.004, 0.14, f"{CLASS_NAMES[cls]}: {n_votes}", ha="left",
                va="center", color=S.INK, fontsize=11.5)
    winner = CLASS_NAMES[int(sum(votes) > len(votes) / 2)]
    ax.text(0.85, 0.14, f"→   Forest says: {winner}", ha="right", va="center", color=S.INK,
            fontsize=12.5, fontweight="bold")
    S.save(fig, OUT / "random-forest-vote.png")
    return votes


if __name__ == "__main__":
    tree_as_questions()
    gini_split()
    tree_depth()
    print("forest votes (1 = flu):", random_forest_vote())
    print("lesson 04 diagrams ->", OUT)
