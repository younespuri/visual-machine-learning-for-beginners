"""Diagrams for lesson 03 - Logistic regression."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("03-logistic-regression")

# The same 20 students the lesson's code uses
HOURS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                  3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0])
PASSED = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                   1, 0, 1, 1, 1, 1, 1, 1, 1, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def class_handles(names, size=9):
    """Legend entries that pair each class color with its marker shape."""
    return [Line2D([], [], ls="", marker=S.MARKERS[i], ms=size, mfc=S.SERIES[i], mec=S.SURFACE,
                   mew=S.RING_W, label=name) for i, name in enumerate(names)]


def clear_spots(ax, cs, points, lo, hi):
    """For each contour line, the spot inside the lo..hi box that sits farthest from every point."""
    pix = ax.transData.transform(points)
    spots = []
    for path in cs.get_paths():
        v = path.vertices
        v = v[(v[:, 0] > lo[0]) & (v[:, 0] < hi[0]) & (v[:, 1] > lo[1]) & (v[:, 1] < hi[1])]
        gap = np.linalg.norm(ax.transData.transform(v)[:, None] - pix[None], axis=2).min(axis=1)
        spots.append(tuple(v[np.argmax(gap)]))
    return spots


def decision_boundary():
    # 70 made-up students: two features, pass/fail drawn from a hidden S-curve
    rng = np.random.default_rng(11)
    n = 70
    hours = rng.uniform(0.3, 9.7, n)
    attend = rng.uniform(42, 100, n)
    passed = (rng.random(n) < sigmoid(1.1 * (hours - 4.6) + 0.11 * (attend - 70))).astype(int)
    points = np.column_stack([hours, attend])
    model = LogisticRegression(max_iter=1000).fit(points, passed)

    fig = S.figure(9.6, 5.6)
    S.header(fig, "Logistic regression splits yes from no with a straight line",
             "Every spot on the map gets a probability: sure far from the line, a coin flip right on it.")
    ax = fig.add_axes([0.075, 0.12, 0.60, 0.67])
    ax.set_xlim(0, 10)
    ax.set_ylim(31, 107)   # empty bands above 100% and below 42% hold the region labels
    gx, gy = np.meshgrid(np.linspace(0, 10, 500), np.linspace(31, 107, 500))
    prob = model.predict_proba(np.column_stack([gx.ravel(), gy.ravel()]))[:, 1].reshape(gx.shape)
    ax.contourf(gx, gy, prob, levels=[0, 0.5, 1], colors=[S.BLUE, S.ORANGE], alpha=S.WASH)
    cs = ax.contour(gx, gy, prob, levels=[0.1, 0.3, 0.7, 0.9], colors=S.MUTED, linewidths=0.9)
    ax.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=9, colors=S.INK_2,
              manual=clear_spots(ax, cs, points, lo=(0.5, 45), hi=(9.5, 97)))
    ax.contour(gx, gy, prob, levels=[0.5], colors=S.INK, linewidths=2.2)
    S.scatter(ax, hours[passed == 0], attend[passed == 0], 0)
    S.scatter(ax, hours[passed == 1], attend[passed == 1], 1)
    ax.text(0.2, 36, "Model predicts: fail", color=S.INK_2, fontsize=10.5, fontweight="bold", va="center")
    ax.text(9.8, 103.5, "Model predicts: pass", color=S.INK_2, fontsize=10.5, fontweight="bold",
            ha="right", va="center")
    ax.set_yticks(range(40, 101, 10))
    ax.set_xlabel("Hours studied per week")
    ax.set_ylabel("Classes attended (%)")
    ax.grid(False)

    handles = class_handles(["Failed (class 0)", "Passed (class 1)"]) + [
        Line2D([], [], color=S.INK, lw=2.2, label="Decision boundary:\nprobability = 50%"),
        Line2D([], [], color=S.MUTED, lw=0.9, label="Other probabilities\nof passing"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.705, 0.79), labelspacing=1.1,
               handlelength=2.2, fontsize=10.5)
    fig.text(0.715, 0.30, "The farther a student sits from\nthe white line, the surer the model.",
             color=S.MUTED, fontsize=10, va="top", linespacing=1.5)
    S.save(fig, OUT / "decision-boundary.png")


def line_vs_s_curve():
    lin = LinearRegression().fit(HOURS.reshape(-1, 1), PASSED)
    log = LogisticRegression().fit(HOURS.reshape(-1, 1), PASSED)
    w, b = log.coef_[0, 0], log.intercept_[0]
    xs = np.linspace(0, 7, 300)

    fig = S.figure(10, 5.6)
    S.header(fig, "Why a straight line fails on yes/no answers",
             "Same 20 students. The line predicts impossible values; the S-curve always stays between 0% and 100%.")
    panels = [("A straight line (linear regression)", lin.predict(xs.reshape(-1, 1))),
              ("An S-curve (logistic regression)", sigmoid(w * xs + b))]
    for i, (name, curve) in enumerate(panels):
        ax = fig.add_axes([0.07 + i * 0.485, 0.135, 0.415, 0.615])
        for lo, hi, word in [(1, 1.45, "above 100%"), (-0.45, 0, "below 0%")]:
            ax.axhspan(lo, hi, color=S.CRITICAL, alpha=0.10, lw=0)
            ax.text(0.15, (lo + hi) / 2, f"✗  impossible: {word}", color=S.INK_2, fontsize=9.5, va="center")
        ax.plot(xs, curve, color=S.INK, lw=2, zorder=2)
        S.scatter(ax, HOURS[PASSED == 0], PASSED[PASSED == 0], 0, label="Failed (0)")
        S.scatter(ax, HOURS[PASSED == 1], PASSED[PASSED == 1], 1, label="Passed (1)")
        ax.set_title(name)
        ax.set_xlim(0, 7)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_xlabel("Hours studied")
        if i == 0:
            ax.set_ylabel("Predicted chance of passing")
            ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.38), fontsize=9.5, handletextpad=0.3)
            continue
        # Right panel: the threshold, the boundary it creates, and three new students
        ax.axhline(0.5, color=S.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(6.95, 0.53, "threshold 0.5", color=S.INK_2, fontsize=9.5, ha="right", va="bottom")
        x0 = -b / w
        ax.plot([x0, x0], [-0.45, 0.5], color=S.MUTED, lw=1.2, ls=(0, (1, 2.5)), zorder=1)
        ax.text(x0 + 0.12, 0.36, f"boundary: {x0:.2f} h", color=S.INK_2, fontsize=9.5, va="center")
        for h, (tx, ty) in zip([1, 3, 5], [(0.1, 0.30), (1.15, 0.74), (5.2, 0.73)]):
            p = sigmoid(w * h + b)
            ax.scatter([h], [p], s=40, marker="D", c=S.INK, edgecolors=S.SURFACE, linewidths=S.RING_W, zorder=4)
            ax.annotate(f"{h} h  →  {p:.0%}", xy=(h, p), xytext=(tx, ty), color=S.INK, fontsize=10.5,
                        va="center", arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkA=2, shrinkB=5))
    S.save(fig, OUT / "line-vs-s-curve.png")


def log_loss_curve():
    p = np.linspace(0.008, 1, 400)
    fig = S.figure(9, 5.2)
    S.header(fig, "Log loss punishes confident mistakes the most",
             "Look at the probability the model gave to the true answer. The closer it is to 0, the steeper the price.")
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.64])
    ax.plot(p, -np.log(p), color=S.INK, lw=2, zorder=2)
    points = [(0.05, S.CRITICAL, "✗  Confident and wrong", (0.12, 3.45)),
              (0.5, S.MUTED, "Unsure", (0.40, 1.80)),
              (0.95, S.GOOD, "✓  Confident and right", (0.73, 1.10))]
    for pv, color, words, (tx, ty) in points:
        loss = -np.log(pv)
        ax.scatter([pv], [loss], s=S.MARK_S, c=color, edgecolors=S.SURFACE, linewidths=S.RING_W, zorder=3)
        ax.annotate(f"{words}\np = {pv:.2f}  →  loss {loss:.2f}", xy=(pv, loss), xytext=(tx, ty),
                    color=S.INK, fontsize=10.5, linespacing=1.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkA=3, shrinkB=6))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 4.9)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Probability the model gave to the true answer")
    ax.set_ylabel("Loss for this example:  −log(p)")
    S.save(fig, OUT / "log-loss.png")


def rounded_cell(ax, x, y, w, h, fc, alpha, ec=None, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, alpha=alpha, ec="none", lw=0, zorder=1))
    if ec:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="none", ec=ec, lw=lw, zorder=2))


def confusion_matrix_explainer():
    fig = S.figure(10.5, 6.2)
    S.header(fig, "The confusion matrix: four ways a yes/no answer can land",
             "Rows are the truth, columns are the model's answer (scikit-learn's layout). Here, positive = spam.")

    # The big 2x2 grid, in data units: each cell is 1 x 1
    ax = fig.add_axes([0.155, 0.05, 0.40, 0.66])
    S.clean_axes(ax)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    cells = {  # (col, row from top): name, short, meaning, example, correct?
        (0, 0): ("True Negative", "TN", "said no, and it was no", "a real email stays\nin your inbox", True),
        (1, 0): ("False Positive", "FP", "said yes, but it was no", "a real email is sent\nto spam: a false alarm", False),
        (0, 1): ("False Negative", "FN", "said no, but it was yes", "spam slips into your\ninbox: a miss", False),
        (1, 1): ("True Positive", "TP", "said yes, and it was yes", "spam is caught:\na hit", True),
    }
    gap = 0.03
    for (c, r), (name, short, meaning, example, ok) in cells.items():
        x, y = c + gap, 1 - r + gap
        rounded_cell(ax, x, y, 1 - 2 * gap, 1 - 2 * gap, S.GOOD if ok else S.CRITICAL, 0.16)
        mark = "✓  correct" if ok else "✗  wrong"
        ax.text(x + 0.07, y + 0.86, mark, color=S.INK_2, fontsize=10, va="center")
        ax.text(x + 0.07, y + 0.66, f"{name} ({short})", color=S.INK, fontsize=12.5, fontweight="bold", va="center")
        ax.text(x + 0.07, y + 0.47, meaning, color=S.INK, fontsize=10.5, va="center")
        ax.text(x + 0.07, y + 0.20, example, color=S.INK_2, fontsize=9.5, va="center", linespacing=1.4)
    for c, label in enumerate(["Predicted: not spam (0)", "Predicted: spam (1)"]):
        ax.text(c + 0.5, 2.06, label, color=S.INK_2, fontsize=10.5, fontweight="bold", ha="center", va="bottom")
    for r, label in enumerate(["Actually:\nnot spam (0)", "Actually:\nspam (1)"]):
        ax.text(-0.06, 1.5 - r, label, color=S.INK_2, fontsize=10.5, fontweight="bold", ha="right", va="center",
                linespacing=1.4)

    # Which cells each score uses: three small grids
    fig.text(0.62, 0.745, "Which cells each score uses", color=S.INK, fontsize=12.5, fontweight="bold", va="bottom")
    scores = [
        ("Accuracy = (TP + TN) / all", "How often is the model right overall?", {(0, 0), (1, 1)}),
        ("Precision = TP / (TP + FP)", "When it says spam, how often is it spam?", {(1, 0), (1, 1)}),
        ("Recall = TP / (TP + FN)", "Of all the spam, how much did it catch?", {(0, 1), (1, 1)}),
    ]
    names = {(0, 0): "TN", (1, 0): "FP", (0, 1): "FN", (1, 1): "TP"}
    for k, (formula, question, used) in enumerate(scores):
        top = 0.66 - k * 0.215
        mini = fig.add_axes([0.62, top - 0.15, 0.085, 0.15])
        S.clean_axes(mini)
        mini.set_xlim(0, 2)
        mini.set_ylim(0, 2)
        for (c, r), short in names.items():
            on = (c, r) in used
            x, y = c + 0.05, 1 - r + 0.05
            rounded_cell(mini, x, y, 0.9, 0.9, S.INK_2 if on else S.PANEL, 0.30 if on else 1.0,
                         ec=S.INK if on else None, lw=1.2)
            mini.text(x + 0.45, y + 0.45, short, color=S.INK if on else S.MUTED, fontsize=9,
                      fontweight="bold" if on else "normal", ha="center", va="center")
        fig.text(0.725, top - 0.045, formula, color=S.INK, fontsize=11.5, fontweight="bold", va="center")
        fig.text(0.725, top - 0.105, question, color=S.INK_2, fontsize=10.5, va="center")
    S.save(fig, OUT / "confusion-matrix.png")


def threshold_gif():
    # 44 made-up emails, scored by a real logistic regression on one feature
    rng = np.random.default_rng(4)
    n0, n1 = 26, 18
    x = np.concatenate([rng.normal(-1.0, 1.0, n0), rng.normal(1.3, 1.0, n1)])
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    prob = LogisticRegression().fit(x.reshape(-1, 1), y).predict_proba(x.reshape(-1, 1))[:, 1]
    jitter = rng.uniform(-0.27, 0.27, y.size)

    def frame(t):
        flagged = prob >= t
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        fn = int(np.sum(~flagged & (y == 1)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn)

        fig = S.figure(10, 5.2)
        S.header(fig, "Moving the threshold trades precision for recall",
                 f"Threshold = {t:.2f}:  every email with P(spam) ≥ {t:.0%} goes to the spam folder.")
        ax = fig.add_axes([0.105, 0.14, 0.52, 0.56])
        ax.axvspan(0, t, color=S.BLUE, alpha=S.WASH, lw=0)
        ax.axvspan(t, 1, color=S.ORANGE, alpha=S.WASH, lw=0)
        ax.axvline(t, color=S.INK, lw=2, zorder=2)
        S.scatter(ax, prob[y == 0], jitter[y == 0], 0, s=60)
        S.scatter(ax, prob[y == 1], 1 + jitter[y == 1], 1, s=64)
        ax.text(t - 0.015, 1.52, "inbox", color=S.INK_2, fontsize=10, ha="right", va="bottom")
        ax.text(t + 0.015, 1.52, "spam folder", color=S.INK_2, fontsize=10, ha="left", va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1], ["Not spam", "Spam"])
        ax.tick_params(axis="y", labelsize=10.5, labelcolor=S.INK_2, length=0)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.set_xlabel("The model's probability of spam, one dot per email")
        ax.grid(axis="y", visible=False)

        side = fig.add_axes([0.68, 0.14, 0.29, 0.62])
        S.clean_axes(side)
        side.set_xlim(0, 1)
        side.set_ylim(0, 1)
        for k, (name, value) in enumerate([("Precision", precision), ("Recall", recall)]):
            yb = 0.86 - k * 0.27
            side.text(0, yb + 0.07, name, color=S.INK, fontsize=12, fontweight="bold", va="bottom")
            side.text(1, yb + 0.07, f"{value:.0%}", color=S.INK, fontsize=12, fontweight="bold",
                      ha="right", va="bottom")
            side.add_patch(Rectangle((0, yb - 0.03), 1, 0.06, fc=S.PANEL, ec="none"))
            side.add_patch(Rectangle((0, yb - 0.03), value, 0.06, fc=S.INK_2, ec="none"))
        side.legend(handles=class_handles(["Not spam", "Spam"], size=8.5), loc="center left",
                    bbox_to_anchor=(-0.03, 0.43), ncol=2, fontsize=10, handletextpad=0.2, columnspacing=1.4)
        side.text(0, 0.30, f"✗  False alarms: {fp:>2}", color=S.INK, fontsize=11, va="center")
        side.text(0, 0.22, "real emails sent to spam", color=S.MUTED, fontsize=9.5, va="center")
        side.text(0, 0.11, f"✗  Missed spam: {fn:>2}", color=S.INK, fontsize=11, va="center")
        side.text(0, 0.03, "spam left in the inbox", color=S.MUTED, fontsize=9.5, va="center")
        fig.text(0.985, 0.018, S.WATERMARK, fontsize=7.5, color=S.MUTED, ha="right", va="bottom")
        img = S.fig_to_image(fig)
        S.plt.close(fig)
        return img

    # 0.5 -> 0.9 (pause) -> 0.1 (pause) -> back to 0.5, so the loop has no jump
    path = np.round(np.r_[np.linspace(0.5, 0.9, 11), [0.9] * 8, np.linspace(0.9, 0.1, 21)[1:], [0.1] * 8,
                          np.linspace(0.1, 0.5, 11)[1:]], 2)
    rendered = {t: frame(t) for t in np.unique(path)}
    return S.save_gif([rendered[t] for t in path], OUT / "threshold.gif", fps=10, hold_last=20)


if __name__ == "__main__":
    decision_boundary()
    line_vs_s_curve()
    log_loss_curve()
    confusion_matrix_explainer()
    threshold_gif()
    print("lesson 03 diagrams ->", OUT)
