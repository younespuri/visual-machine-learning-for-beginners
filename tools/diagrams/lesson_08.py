"""Diagrams for lesson 08 - Overfitting and cross-validation."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("08-overfitting-and-cross-validation")

# The lesson's temperature data: 40 random days, the same seed as in "Code it"
SEED = 897
MONTHS = ([0, 91, 182, 274], ["Jan", "Apr", "Jul", "Oct"])
TRUE_STYLE = dict(color=S.MUTED, lw=1.2, ls=(0, (4, 3)))


def season(day):
    """The true seasonal pattern: coldest around January 20, warmest in July."""
    return 12 - 10 * np.cos(2 * np.pi * (day - 20) / 365)


def temperature_data():
    """The exact 20 training days and 20 test days from the lesson's code."""
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(SEED)
    day = rng.uniform(0, 365, 40)
    temp = season(day) + rng.normal(0, 3, 40)
    return train_test_split(day, temp, test_size=0.5, random_state=0)


def fit(day, temp, degree):
    return np.polynomial.Polynomial.fit(day, temp, deg=degree)


def mse(y, y_pred):
    return float(np.mean((y - y_pred) ** 2))


def month_axis(ax):
    ax.set_xticks(*MONTHS)
    ax.set_xlim(0, 365)


def legend_handle(cls, label, **kw):
    size = 6.5 if S.MARKERS[cls] == "s" else 7
    return Line2D([], [], color=S.SERIES[cls], marker=S.MARKERS[cls], ls="", markersize=size,
                  label=label, **kw)


def verdict(ax, ok, text, y=1.17, size=12.5):
    """A bold panel title led by a check or a cross (never color alone)."""
    mark, col = ("✓", S.GOOD) if ok else ("✗", S.CRITICAL)
    ax.text(0, y, mark, color=col, fontsize=size + 0.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.075, y, text, color=S.INK, fontsize=size, fontweight="bold", transform=ax.transAxes)


def centered_verdict(fig, cx, y, ok, text, size=12):
    """Like verdict(), but centred on cx: the mark sits just left of the measured title."""
    mark, col = ("✓", S.GOOD) if ok else ("✗", S.CRITICAL)
    t = fig.text(cx + 0.011, y, text, ha="center", color=S.INK, fontsize=size, fontweight="bold")
    fig.canvas.draw()
    x0 = t.get_window_extent().transformed(fig.transFigure.inverted()).x0
    fig.text(x0 - 0.007, y, mark, ha="right", color=col, fontsize=size + 0.5, fontweight="bold")


# 1 · Hero: the same data, fitted three ways ---------------------------------
def triptych():
    d_tr, d_te, t_tr, t_te = temperature_data()
    panels = [(1, "Underfit", False, "too simple: misses the seasons"),
              (4, "Just right", True, "learns the pattern, not the noise"),
              (15, "Overfit", False, "memorizes the noise")]
    fig = S.figure(10.5, 5.4)
    S.header(fig, "Too simple, just right, too complex",
             "Three curves learn the seasons from the same 20 days. New days they never saw reveal the winner.")
    xs = np.linspace(0, 365, 800)
    for i, (deg, name, ok, note) in enumerate(panels):
        ax = fig.add_axes([0.07 + i * 0.315, 0.21, 0.27, 0.49])
        curve = fit(d_tr, t_tr, deg)
        ax.plot(xs, season(xs), zorder=1, **TRUE_STYLE)
        ax.plot(xs, curve(xs), color=S.ORANGE, zorder=2)
        S.scatter(ax, d_tr, t_tr, 0, s=46)
        S.scatter(ax, d_te, t_te, 2, s=40)
        month_axis(ax)
        ax.set_ylim(-8, 33)
        if i == 0:
            ax.set_ylabel("Temperature (°C)")
        else:
            ax.set_yticklabels([])
        verdict(ax, ok, f"{name} · degree {deg}")
        ax.text(0, 1.05, f"train MSE {mse(t_tr, curve(d_tr)):.1f}   ·   test MSE {mse(t_te, curve(d_te)):.1f}",
                color=S.INK_2, fontsize=10, transform=ax.transAxes)
        ax.text(0.5, -0.19, note, color=S.INK_2, fontsize=10, ha="center", transform=ax.transAxes)
    handles = [legend_handle(0, "training days"), legend_handle(2, "test days (never seen)"),
               Line2D([], [], color=S.ORANGE, label="model's curve"),
               Line2D([], [], label="true pattern", **TRUE_STYLE)]
    fig.legend(handles=handles, loc="lower left", ncol=4, bbox_to_anchor=(0.03, 0.0), columnspacing=1.2)
    S.save(fig, OUT / "too-simple-just-right-too-complex.png")


# 2 · Error vs model complexity -------------------------------------------------
def typical_errors(repeats=2000, degrees=range(1, 16)):
    """Median train / validation MSE over many random samples of 20 + 20 days."""
    rng = np.random.default_rng(8)
    train, valid = [], []
    for _ in range(repeats):
        d_tr = rng.uniform(0, 365, 20)
        t_tr = season(d_tr) + rng.normal(0, 3, 20)
        d_va = rng.uniform(0, 365, 20)
        t_va = season(d_va) + rng.normal(0, 3, 20)
        curves = [fit(d_tr, t_tr, d) for d in degrees]
        train.append([mse(t_tr, c(d_tr)) for c in curves])
        valid.append([mse(t_va, c(d_va)) for c in curves])
    return np.median(train, axis=0), np.median(valid, axis=0)


def error_vs_complexity():
    degrees = np.arange(1, 16)
    train, valid = typical_errors(degrees=degrees)
    best = int(degrees[np.argmin(valid)])
    top = 70

    fig = S.figure(9.5, 5.6)
    S.header(fig, "Training error always falls. Validation error doesn't.",
             "Typical errors over 2,000 random samples of 20 days, for every polynomial degree from 1 to 15.")
    ax = fig.add_axes([0.085, 0.13, 0.86, 0.655])
    ax.plot(degrees, train, color=S.BLUE, marker=S.MARKERS[0], markersize=6,
            markeredgecolor=S.SURFACE, markeredgewidth=1.2, label="training error", zorder=3)
    ok = valid <= top
    exit_i = int(np.argmax(~ok))                     # first degree that leaves the chart
    ax.plot(degrees[ok], valid[ok], color=S.ORANGE, marker=S.MARKERS[1], markersize=7,
            markeredgecolor=S.SURFACE, markeredgewidth=1.2, label="validation error", zorder=3)
    # the validation line keeps rising past the top edge
    x0, y0, x1, y1 = degrees[exit_i - 1], valid[exit_i - 1], degrees[exit_i], valid[exit_i]
    xe = x0 + (top - y0) / (y1 - y0) * (x1 - x0)
    ax.plot([x0, xe], [y0, top], color=S.ORANGE, zorder=2, clip_on=False)
    ax.annotate(f"keeps climbing:\n{valid[exit_i]:,.0f} at degree {x1},\nmillions by degree 15",
                xy=(xe + 0.05, top - 1.5), xytext=(xe + 0.7, top - 3), color=S.INK_2, fontsize=10, va="top",
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1))

    # the sweet spot, and the two zones on either side of it
    ax.axvline(best, color=S.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.scatter([best], [valid[best - 1]], s=260, facecolors="none", edgecolors=S.INK, linewidths=1.4, zorder=4)
    ax.annotate(f"sweet spot: degree {best}\nlowest validation error", xy=(best, valid[best - 1]),
                xytext=(best + 0.6, 34), color=S.INK, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1))
    ax.text(best - 0.2, top - 3, "← underfitting\nboth errors high", color=S.MUTED, fontsize=10,
            va="top", ha="right", multialignment="right")
    ax.text(best + 0.2, top - 3, "overfitting →\nthe gap grows", color=S.MUTED, fontsize=10, va="top")

    # the gap between the two curves
    g = 7
    ax.annotate("", xy=(g + 0.12, valid[g - 1] - 0.8), xytext=(g + 0.12, train[g - 1] + 0.8),
                arrowprops=dict(arrowstyle="<->", color=S.INK_2, lw=1.1))
    ax.text(g + 0.3, (valid[g - 1] + train[g - 1]) / 2, "the gap:\noverfitting", color=S.INK_2,
            fontsize=10, va="center")

    # direct labels next to the lines
    ax.text(degrees[-1] + 0.25, train[-1], "training", color=S.INK_2, fontsize=10, va="center")
    ax.text(8.45, 30, "validation", color=S.INK_2, fontsize=10, va="center")

    ax.set_xlim(0.5, 16.3)
    ax.set_ylim(0, top)
    ax.set_xticks(degrees)
    ax.set_xlabel("Model complexity (polynomial degree)")
    ax.set_ylabel("Error (MSE)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
    S.save(fig, OUT / "error-vs-complexity.png")


# 3 · Bias and variance ---------------------------------------------------------
def target(ax, hits):
    for r, fc in zip([1.0, 0.75, 0.5, 0.25], [S.PANEL, S.SURFACE, S.PANEL, S.SURFACE]):
        ax.add_patch(Circle((0, 0), r, fc=fc, ec=S.AXIS, lw=1.1, zorder=1))
    ax.add_patch(Circle((0, 0), 0.1, fc=S.INK_2, ec="none", alpha=0.55, zorder=1))
    S.scatter(ax, hits[:, 0], hits[:, 1], 1, s=46)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    S.clean_axes(ax)


def bias_and_variance(n_models=20):
    rng = np.random.default_rng(4)
    shots = [rng.normal([-0.42, 0.38], 0.07, (12, 2)),     # grouped, but off-centre
             rng.normal([0.0, 0.0], 0.08, (12, 2)),        # grouped on the bullseye
             rng.normal([0.0, 0.0], 0.42, (12, 2))]        # centred on average, scattered
    shots[2] = np.clip(shots[2], -0.98, 0.98)

    samples = []
    for _ in range(n_models):
        d = rng.uniform(0, 365, 20)
        samples.append((d, season(d) + rng.normal(0, 3, 20)))

    cols = [(1, "Too simple · degree 1", False, "High bias, low variance",
             "Every model misses in the same way"),
            (4, "Just right · degree 4", True, "Low bias, low variance",
             "Models agree, and follow the truth"),
            (15, "Too complex · degree 15", False, "Low bias, high variance",
             "Models disagree wildly")]
    fig = S.figure(10.5, 6.9)
    S.header(fig, "Bias and variance: retrain the same model 20 times",
             "Top: the archery picture. Bottom: 20 real models, each trained on its own random 20 days.")
    xs = np.linspace(0, 365, 600)
    for i, (deg, title, ok, kind, note) in enumerate(cols):
        left = 0.07 + i * 0.315
        cx = left + 0.135
        centered_verdict(fig, cx, 0.795, ok, title)
        tax = fig.add_axes([cx - 0.1, 0.535, 0.2, 0.235])
        target(tax, shots[i])
        fig.text(cx, 0.515, kind, color=S.INK_2, fontsize=10.5, ha="center", va="top")

        ax = fig.add_axes([left, 0.155, 0.27, 0.29])
        for d, t in samples:
            ax.plot(xs, fit(d, t, deg)(xs), color=S.ORANGE, lw=1.0, alpha=0.5, zorder=2)
        ax.plot(xs, season(xs), color=S.INK, lw=1.6, ls=(0, (4, 3)), zorder=3)
        month_axis(ax)
        ax.set_ylim(-8, 33)
        if i == 0:
            ax.set_ylabel("Temperature (°C)")
        else:
            ax.set_yticklabels([])
        ax.text(0.5, -0.22, note, color=S.INK_2, fontsize=10, ha="center", transform=ax.transAxes)
    handles = [Line2D([], [], color=S.ORANGE, marker=S.MARKERS[1], ls="", markersize=7,
                      label="one model's guess (an arrow)"),
               Line2D([], [], color=S.ORANGE, lw=1.2, label="one model's curve"),
               Line2D([], [], color=S.INK, lw=1.6, ls=(0, (4, 3)), label="the truth (the bullseye)")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.03, 0.0), columnspacing=1.8)
    S.save(fig, OUT / "bias-and-variance.png")


# 4 · k-fold cross-validation ---------------------------------------------------
def fold_scores():
    """The exact 5 fold scores from the lesson's cross_val_score step."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_redundant=1,
                               n_clusters_per_class=2, flip_y=0.08, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=42), X_train, y_train, cv=kf)


def block(ax, x, y, w, h, fc, text, alpha=1.0, ec=None, lw=0, color=S.INK, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                fc=fc, ec=ec or fc, lw=lw, alpha=alpha, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size,
            fontweight=weight, transform=ax.transAxes, zorder=3)


def k_fold():
    scores = fold_scores()
    k = len(scores)
    fig = S.figure(10, 5.8)
    S.header(fig, "5-fold cross-validation: five exams instead of one",
             "Each round trains on 4 folds and validates on the fifth. Every row of data is validated exactly once.")
    ax = fig.add_axes([0, 0, 1, 1])
    S.clean_axes(ax)
    ax.patch.set_alpha(0)

    x0, fw, gap, bh = 0.13, 0.108, 0.006, 0.085      # fold grid geometry (figure fractions)
    rows_top = 0.66
    ax.text(x0 + (k * fw) / 2, 0.755, "Training data: 150 rows, split into 5 folds of 30",
            ha="center", color=S.INK_2, fontsize=10.5, transform=ax.transAxes)
    for r in range(k):
        y = rows_top - r * (bh + 0.022)
        ax.text(x0 - 0.018, y + bh / 2, f"Round {r + 1}", ha="right", va="center", color=S.INK_2,
                fontsize=10.5, transform=ax.transAxes)
        for f in range(k):
            is_val = f == r
            block(ax, x0 + f * fw + gap / 2, y, fw - gap, bh,
                  S.ORANGE if is_val else S.BLUE, "validate" if is_val else "train",
                  alpha=1.0 if is_val else 0.55, weight="bold" if is_val else "normal")
        ax.text(x0 + k * fw + 0.03, y + bh / 2, "→", ha="left", va="center", color=S.MUTED,
                fontsize=13, transform=ax.transAxes)
        ax.text(x0 + k * fw + 0.065, y + bh / 2, f"{scores[r]:.3f}", ha="left", va="center",
                color=S.INK, fontsize=11.5, transform=ax.transAxes)
    sx = x0 + k * fw + 0.065
    ax.text(sx, 0.755, "score", color=S.INK_2, fontsize=10.5, transform=ax.transAxes)
    y_last = rows_top - (k - 1) * (bh + 0.022)
    ax.plot([sx - 0.005, sx + 0.075], [y_last - 0.025, y_last - 0.025], color=S.MUTED, lw=1,
            transform=ax.transAxes)
    ax.text(sx, y_last - 0.06, f"mean {scores.mean():.3f}", color=S.INK, fontsize=12,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(sx, y_last - 0.105, f"std {scores.std():.3f}", color=S.INK_2, fontsize=10.5,
            va="center", transform=ax.transAxes)

    # the locked test set
    tx, tw = 0.8, 0.15
    ty = y_last
    th = rows_top + bh - ty
    block(ax, tx, ty, tw, th, S.AQUA, "", alpha=0.18, ec=S.AQUA, lw=1.4)
    ax.text(tx + tw / 2, ty + th * 0.62, "Test set", ha="center", va="center", color=S.INK,
            fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(tx + tw / 2, ty + th * 0.4, "50 rows\nlocked away\nuntil the very end", ha="center",
            va="center", color=S.INK_2, fontsize=10, linespacing=1.4, transform=ax.transAxes)

    handles = [Patch(fc=to_rgba(S.BLUE, 0.55), ec="none", label="training folds"),
               Patch(fc=S.ORANGE, ec="none", label="validation fold"),
               Patch(fc=to_rgba(S.AQUA, 0.18), ec=S.AQUA, lw=1.2, label="test set (not used here)")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.03, 0.0), columnspacing=1.8,
               handlelength=1.6, handleheight=1.1)
    S.save(fig, OUT / "k-fold-cross-validation.png")


# 5 · GIF: complexity sweeping from degree 1 to 15 ------------------------------
def complexity_sweep_gif(hold=7, morph=4):
    d_tr, d_te, t_tr, t_te = temperature_data()
    degrees = np.arange(1, 16)
    curves = [fit(d_tr, t_tr, d) for d in degrees]
    train = np.array([mse(t_tr, c(d_tr)) for c in curves])
    test = np.array([mse(t_te, c(d_te)) for c in curves])
    xs = np.linspace(0, 365, 600)
    ys = [c(xs) for c in curves]

    def frame(i, t):
        """Degree index i, blended a fraction t of the way toward degree i + 1."""
        e = t * t * (3 - 2 * t)                       # ease in and out
        y_curve = ys[i] if t == 0 else (1 - e) * ys[i] + e * ys[i + 1]
        fig = S.figure(9, 5)
        S.header(fig, "From underfit to overfit, one degree at a time",
                 f"Degree {degrees[i]:>2}:    train MSE = {train[i]:5.1f}     test MSE = {test[i]:5.1f}")
        ax = fig.add_axes([0.075, 0.2, 0.47, 0.57])
        ax.plot(xs, season(xs), zorder=1, **TRUE_STYLE)
        ax.plot(xs, y_curve, color=S.ORANGE, zorder=2)
        S.scatter(ax, d_tr, t_tr, 0, s=40)
        S.scatter(ax, d_te, t_te, 2, s=34)
        month_axis(ax)
        ax.set_ylim(-8, 33)
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("The curve bends more and more")
        day_handles = [legend_handle(0, "training days"), legend_handle(2, "test days"),
                       Line2D([], [], label="true pattern", **TRUE_STYLE)]
        ax.legend(handles=day_handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9.5)

        ax2 = fig.add_axes([0.63, 0.2, 0.33, 0.57])
        n = i + 1
        xd = list(degrees[:n]) + ([degrees[i] + t] if t else [])
        tr = list(train[:n]) + ([train[i] + t * (train[i + 1] - train[i])] if t else [])
        te = list(test[:n]) + ([test[i] + t * (test[i + 1] - test[i])] if t else [])
        ax2.plot(xd, tr, color=S.BLUE)
        ax2.plot(xd, te, color=S.AQUA)
        S.scatter(ax2, degrees[:n], train[:n], 0, s=26)
        S.scatter(ax2, degrees[:n], test[:n], 2, s=22)
        ax2.set_xlim(0.5, 15.5)
        ax2.set_ylim(0, 70)
        ax2.set_xticks([1, 4, 8, 12, 15])
        ax2.set_xlabel("degree")
        ax2.set_ylabel("MSE (error)")
        ax2.set_title("...and the errors split apart")
        err_handles = [Line2D([], [], color=S.BLUE, marker="o", markersize=5, label="train error"),
                       Line2D([], [], color=S.AQUA, marker="s", markersize=4.5, label="test error")]
        ax2.legend(handles=err_handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), fontsize=9.5)
        fig.text(0.985, 0.018, S.WATERMARK, fontsize=7.5, color=S.MUTED, ha="right", va="bottom")
        img = S.fig_to_image(fig)
        S.plt.close(fig)
        return img

    frames = []
    for i in range(len(degrees)):
        still = frame(i, 0)
        frames += [still] * (hold + (6 if i in (0, 3) else 0))
        if i < len(degrees) - 1:
            frames += [frame(i, (m + 1) / (morph + 1)) for m in range(morph)]
    return S.save_gif(frames, OUT / "complexity-sweep.gif", fps=12)


if __name__ == "__main__":
    triptych()
    error_vs_complexity()
    bias_and_variance()
    k_fold()
    complexity_sweep_gif()
    print("lesson 08 diagrams ->", OUT)
