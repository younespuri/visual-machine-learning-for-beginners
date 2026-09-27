"""Diagrams for lesson 06 - Naive Bayes and unsupervised learning."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("06-naive-bayes-and-unsupervised-learning")

HAM, SPAM = 0, 1          # series slots: not spam = BLUE circle, spam = ORANGE triangle


def canvas(fig):
    """A full-figure axes measured in layout pixels (100 dpi), with y pointing down."""
    w, h = fig.get_size_inches() * fig.dpi
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_facecolor("none")
    S.clean_axes(ax)
    return ax


def hbar(ax, x, y, length, color, thick=20, radius=4, zorder=2):
    """A horizontal bar: square at the baseline, 4px rounded data end."""
    if length <= 0:
        return
    r = min(radius, length / 2, thick / 2)
    top = y - thick / 2
    ax.add_patch(FancyBboxPatch((x, top), length, thick, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec="none", lw=0, zorder=zorder))
    ax.add_patch(Rectangle((x, top), length - r, thick, fc=color, ec="none", lw=0, zorder=zorder))


def card(ax, x, y, w, h, ec=S.AXIS, lw=1.0, radius=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=S.PANEL, ec=ec, lw=lw, zorder=2))


def spam_evidence():
    fig = S.figure(10, 5.4)
    S.header(fig, "Naive Bayes weighs every word as evidence",
             "It starts from how common spam is, then each word nudges the odds up or down.")
    ax = canvas(fig)

    steps = [  # card title, card note, how much more likely the word is in spam than in normal email
        ("Before reading", "4 in 10 emails are spam", None),
        ("“free”", "↑  7× more common in spam", 7),
        ("“winner”", "↑  20× more common in spam", 20),
        ("“meeting”", "↓  10× more common in normal email", 0.1),
    ]
    odds, probs = 0.4 / 0.6, []
    for *_, ratio in steps:
        odds *= ratio or 1
        probs.append(odds / (1 + odds))

    card_x, card_w, card_h = 35, 285, 56
    bar_x, bar_len = 360, 500
    row_y = [176 + 72 * i for i in range(len(steps))]
    head_y = 124

    ax.text(card_x, head_y, "What the filter reads", color=S.MUTED, fontsize=10, va="center")
    ax.scatter([bar_x + 7], [head_y + 1], s=S.MARK_S, c=S.SERIES[SPAM], marker=S.MARKERS[SPAM],
               edgecolors=S.SURFACE, linewidths=S.RING_W, zorder=3)
    ax.text(bar_x + 22, head_y, "Chance the email is spam", color=S.MUTED, fontsize=10, va="center")

    # the 50% decision line: above it, the email goes to the spam folder
    mid = bar_x + bar_len / 2
    ax.plot([mid, mid], [row_y[0] - 26, row_y[-1] + 28], color=S.MUTED, lw=1, zorder=1)
    ax.text(mid, row_y[0] - 36, "50%", color=S.MUTED, fontsize=9, ha="center", va="center")
    below = row_y[-1] + 44

    for i, ((title, note, _), p, y) in enumerate(zip(steps, probs, row_y)):
        card(ax, card_x, y - card_h / 2, card_w, card_h)
        ax.text(card_x + 16, y - 10, title, color=S.INK, fontsize=12.5, fontweight="bold", va="center")
        ax.text(card_x + 16, y + 13, note, color=S.INK_2, fontsize=10, va="center")
        hbar(ax, bar_x, y, bar_len, S.PANEL, thick=22, zorder=1)
        hbar(ax, bar_x, y, bar_len * p, S.SERIES[SPAM], thick=22)
        if i:  # a tick where the belief was before this word
            px = bar_x + bar_len * probs[i - 1]
            ax.plot([px, px], [y - 16, y + 16], color=S.INK, lw=1.5, zorder=4)
        ax.text(bar_x + bar_len + 18, y, f"{p:.0%}", color=S.INK, fontsize=13, fontweight="bold",
                va="center")

    key_x = bar_x + bar_len
    ax.text(key_x, below, "where the belief was before this word", color=S.MUTED, fontsize=9,
            ha="right", va="center")
    ax.plot([key_x - 222, key_x - 222], [below - 8, below + 8], color=S.INK, lw=1.5)

    y_end = below + 50
    ax.scatter([card_x + 9], [y_end + 1], s=90, c=S.SERIES[SPAM], marker=S.MARKERS[SPAM],
               edgecolors=S.SURFACE, linewidths=S.RING_W, zorder=3)
    ax.text(card_x + 26, y_end, f"Verdict: spam, with {probs[-1]:.0%} probability", color=S.INK,
            fontsize=12.5, fontweight="bold", va="center")
    S.save(fig, OUT / "spam-evidence.png")


def bayes_counting():
    fig = S.figure(10, 5.8)
    S.header(fig, "Bayes' rule is just counting inside a smaller group",
             "Of 100 emails, 40 are spam. Keep only the 34 that contain “free”: 28 of those are spam.")
    ax = canvas(fig)
    cell, m = 30, 4

    def grid(x0, y0, rows, cols, n=None):
        pts = [(x0 + cell * (c + 0.5), y0 + cell * (r + 0.5), SPAM if r < 4 else HAM)
               for r in range(rows) for c in range(cols)][:n]
        for cls in (HAM, SPAM):
            S.scatter(ax, [p[0] for p in pts if p[2] == cls], [p[1] for p in pts if p[2] == cls],
                      cls, s=64)

    def free_region(x0, y0, spam_cols, ham_cols):
        """Outline the emails that contain "free": spam_cols of each spam row + ham_cols in row 5."""
        v = [(x0 - m, y0 - m), (x0 + spam_cols * cell + m, y0 - m),
             (x0 + spam_cols * cell + m, y0 + 4 * cell), (x0 + ham_cols * cell + m, y0 + 4 * cell),
             (x0 + ham_cols * cell + m, y0 + 5 * cell + m), (x0 - m, y0 + 5 * cell + m)]
        ax.add_patch(Polygon(v, closed=True, fc=S.INK, alpha=0.07, ec="none", zorder=1))
        ax.add_patch(Polygon(v, closed=True, fc="none", ec=S.INK_2, lw=1.3, joinstyle="round",
                             zorder=1))

    top = 174
    # left: all 100 emails. Spam = top 4 rows; "free" = 7 of each spam row + 6 normal emails.
    gx = 150
    free_region(gx, top, 7, 6)
    grid(gx, top, 10, 10)
    ax.text(gx - m, top - 28, "All 100 emails", color=S.INK, fontsize=12.5, fontweight="bold",
            va="center")
    for r0, r1, label in [(0, 4, "40 spam"), (4, 10, "60 not spam")]:
        ya, yb = top + r0 * cell + 4, top + r1 * cell - 4
        ax.plot([gx - 16, gx - 16], [ya, yb], color=S.MUTED, lw=1.2)
        ax.text(gx - 26, (ya + yb) / 2, label, color=S.INK_2, fontsize=10, ha="right", va="center")
    stat_y = top + 10 * cell + 30
    ax.text(gx - m, stat_y, "P(spam) = 40 / 100 = 40%", color=S.INK, fontsize=12.5,
            fontweight="bold", va="center")

    # right: only the 34 emails that contain "free", vertically centred on the left grid
    rx, ry = 640, top + 2.5 * cell
    free_region(rx, ry, 7, 6)
    grid(rx, ry, 5, 7, n=34)
    ax.text(rx - m, top - 28, "Only the 34 emails with “free”", color=S.INK, fontsize=12.5,
            fontweight="bold", va="center")
    ax.text(rx - m, stat_y, "P(spam | free) = 28 / 34 = 82%", color=S.INK, fontsize=12.5,
            fontweight="bold", va="center")
    ax.text(rx - m, stat_y + 26, "Bayes' rule agrees:  0.7 × 0.4 / 0.34 ≈ 0.82", color=S.INK_2,
            fontsize=10.5, va="center")

    ay = ry + 2.5 * cell
    ax.annotate("", xy=(rx - 22, ay), xytext=(gx + 10 * cell + 22, ay),
                arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.6, mutation_scale=14))
    ax.text((gx + 10 * cell + rx) / 2, ay - 30, "keep only emails\nthat contain “free”",
            color=S.INK_2, fontsize=10, ha="center", va="center", linespacing=1.4)

    handles = [Line2D([], [], ls="none", marker=S.MARKERS[SPAM], color=S.SERIES[SPAM], ms=9,
                      markeredgecolor=S.SURFACE, label="spam"),
               Line2D([], [], ls="none", marker=S.MARKERS[HAM], color=S.SERIES[HAM], ms=9,
                      markeredgecolor=S.SURFACE, label="not spam"),
               Patch(fc=(1, 1, 1, 0.07), ec=S.INK_2, lw=1.3, label="contains “free”")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.845), ncol=3,
               handletextpad=0.5, columnspacing=1.6)
    S.save(fig, OUT / "bayes-counting.png")


def smoothing():
    fig = S.figure(10, 4.75)
    S.header(fig, "Smoothing: one unseen word must not erase the evidence",
             "“meeting” never showed up in the training spam, so its raw probability is 0,"
             " and one 0 wipes out the whole product.")
    ax = canvas(fig)

    words = ["claim", "free", "prize", "meeting"]
    spam_counts = [4, 4, 2, 0]           # from the lesson's 6 training spam messages
    total, vocab = 31, 45
    smoothed = np.prod([(c + 1) / (total + vocab) for c in spam_counts])
    rows = [
        ("Raw counts", "count / 31", [f"{c} / {total}" for c in spam_counts], "0", False,
         "the spammy words now count for nothing"),
        ("Add-one smoothing", "(count + 1) / (31 + 45)",
         [f"{c + 1} / {total + vocab}" for c in spam_counts], f"{smoothed:.7f}", True,
         "tiny, but every word still counts"),
    ]
    chip_w, chip_h, step = 124, 60, 160
    x0 = 35
    for k, (name, rule, fracs, result, ok, verdict) in enumerate(rows):
        y = 122 + k * 148
        ax.text(x0, y, name, color=S.INK, fontsize=12.5, fontweight="bold", va="center")
        ax.text(x0 + (112 if k == 0 else 178), y, rule, color=S.MUTED, fontsize=10.5, va="center")
        top = y + 22
        for i, (w, f) in enumerate(zip(words, fracs)):
            x = x0 + i * step
            zero = not ok and w == "meeting"
            card(ax, x, top, chip_w, chip_h, ec=S.CRITICAL if zero else S.AXIS, lw=1.8 if zero else 1.0)
            ax.text(x + chip_w / 2, top + 19, f"P({w} | spam)", color=S.INK_2, fontsize=9.5,
                    ha="center", va="center")
            ax.text(x + chip_w / 2, top + 41, f, color=S.INK, fontsize=14, fontweight="bold",
                    ha="center", va="center")
            if zero:
                ax.text(x + chip_w / 2, top + chip_h + 18, "never seen in spam", color=S.INK_2,
                        fontsize=9.5, ha="center", va="center")
            ax.text(x + chip_w + (step - chip_w) / 2, top + chip_h / 2,
                    "×" if i < len(words) - 1 else "=", color=S.MUTED, fontsize=16,
                    ha="center", va="center")
        rx, rw = x0 + len(words) * step, 170
        color = S.GOOD if ok else S.CRITICAL
        card(ax, rx, top, rw, chip_h, ec=color, lw=1.8)
        ax.text(rx + rw / 2, top + 19, "spam score", color=S.INK_2, fontsize=9.5, ha="center",
                va="center")
        ax.text(rx + rw / 2, top + 41, result, color=S.INK, fontsize=14, fontweight="bold",
                ha="center", va="center")
        ax.text(rx, top + chip_h + 18, "✓" if ok else "✗", color=color, fontsize=13,
                fontweight="bold", va="center")
        ax.text(rx + 18, top + chip_h + 18, verdict, color=S.INK_2, fontsize=9.5, va="center")

    ax.text(x0, 424, "31 = words in the training spam    ·    45 = different words in the vocabulary"
            "    ·    in scikit-learn, the +1 is MultinomialNB(alpha=1.0), the default",
            color=S.MUTED, fontsize=9.5, va="center")
    S.save(fig, OUT / "smoothing.png")


def clip_halfplane(poly, point, normal):
    """Keep the part of a convex polygon where (p - point) . normal >= 0 (Sutherland-Hodgman)."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = np.dot(a - point, normal), np.dot(b - point, normal)
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + da / (da - db) * (b - a))
    return np.array(out)


def group_outline(ax, pts, center, others, pad_px=9, gap_px=0):
    """A rounded hull around one k-means group, cut back so it never crosses a neighbour's border.

    Works in screen pixels, so the padding comes out round whatever the axis scales are.
    On iris the two big groups almost touch (their closest flowers sit ~2px from the border),
    so the cut runs exactly along the border: every point stays inside its own outline.
    """
    from scipy.spatial import ConvexHull

    T = ax.transData
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) * pad_px
    cloud = (T.transform(pts)[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    poly = cloud[ConvexHull(cloud).vertices]
    for other in others:
        # k-means borders: the points equally far from two group centres (a straight line)
        mid, d = (center + other) / 2, center - other
        p0, p1 = T.transform([mid, mid + np.array([-d[1], d[0]])])
        normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
        normal /= np.linalg.norm(normal)
        if np.dot(T.transform([center])[0] - p0, normal) < 0:
            normal = -normal
        poly = clip_halfplane(poly, p0 + gap_px * normal, normal)
    return T.inverted().transform(poly)


def supervised_vs_unsupervised():
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data[:, 2:4], iris.target
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    order = np.argsort(km.cluster_centers_[:, 0])          # number the groups left to right

    fig = S.figure(10, 5.4)
    S.header(fig, "Same flowers, two different jobs",
             "Supervised learning is given the answers. Unsupervised learning gets only the data"
             " and looks for groups.")
    lims = dict(xlim=(0.4, 7.4), ylim=(-0.25, 2.85))

    ax = fig.add_axes([0.07, 0.12, 0.4, 0.6])
    for cls, name in enumerate(iris.target_names):
        S.scatter(ax, X[y == cls, 0], X[y == cls, 1], cls, label=name, s=52)
    ax.set_title("Supervised: every flower has a label")
    ax.legend(loc="upper left", handletextpad=0.3)
    ax.set(xlabel="Petal length (cm)", ylabel="Petal width (cm)", **lims)

    ax2 = fig.add_axes([0.56, 0.12, 0.4, 0.6])
    ax2.set(xlabel="Petal length (cm)", **lims)
    ax2.set_title("Unsupervised: no labels, just structure")
    ax2.scatter(X[:, 0], X[:, 1], s=52, c=S.MUTED, marker="o", edgecolors=S.SURFACE,
                linewidths=S.RING_W, zorder=3)
    label_at = [((2.25, 0.3), "left"), ((2.85, 1.45), "right"), ((4.55, 2.5), "right")]
    centers = km.cluster_centers_
    for n, (k, (xy, ha)) in enumerate(zip(order, label_at), start=1):
        others = [c for j, c in enumerate(centers) if j != k]
        outline = group_outline(ax2, X[km.labels_ == k], centers[k], others)
        ax2.add_patch(Polygon(outline, closed=True, fc=S.INK, alpha=0.06, ec="none", zorder=1))
        ax2.add_patch(Polygon(outline, closed=True, fc="none", ec=S.INK_2, lw=1.2, zorder=2))
        ax2.text(*xy, f"group {n}", color=S.INK_2, fontsize=10.5, ha=ha, va="center")
    S.save(fig, OUT / "supervised-vs-unsupervised.png")


if __name__ == "__main__":
    spam_evidence()
    bayes_counting()
    smoothing()
    supervised_vs_unsupervised()
    print("lesson 06 diagrams ->", OUT)
