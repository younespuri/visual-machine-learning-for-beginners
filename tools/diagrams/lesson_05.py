"""Diagrams for lesson 05 - KNN and SVM."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("05-knn-and-svm")

# The new customer from the lesson's code: [age, yearly income in dollars]
CUSTOMER_A = np.array([25, 50_000])


def customers():
    """The exact reading-glasses customers of the lesson's code (Steps 2 and 3)."""
    rng = np.random.default_rng(34)
    n = 300
    bought = np.repeat([1, 0], n // 2)
    age = np.where(bought == 1, rng.normal(52, 8, n), rng.normal(32, 8, n))
    age = age.clip(18, 80).round().astype(int)
    income = rng.uniform(20_000, 120_000, n).round(-2).astype(int)
    X = np.column_stack([age, income])
    return train_test_split(X, bought, test_size=0.25, random_state=42, stratify=bought)


def legend_handles(labels, extra=()):
    """Legend entries for the classes (color + shape), plus optional extra handles."""
    handles = [Line2D([], [], ls="", marker=S.MARKERS[i], ms=8.5, mfc=S.SERIES[i], mec=S.SURFACE,
                      mew=1.0, label=lab) for i, lab in enumerate(labels)]
    return handles + list(extra)


def new_point_handle(label):
    return Line2D([], [], ls="", marker="*", ms=15, mfc=S.INK, mec=S.SURFACE, mew=1.0, label=label)


def ring_handle(label):
    return Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec=S.INK, mew=1.3, label=label)


def new_point(ax, x, y, s=330):
    ax.scatter([x], [y], s=s, marker="*", c=S.INK, edgecolors=S.SURFACE, linewidths=1.2, zorder=6)


def ring(ax, x, y, s=300):
    ax.scatter(x, y, s=s, facecolors="none", edgecolors=S.INK, linewidths=1.3, zorder=5)


def knn_vote():
    """Hero: a new point, its 5 nearest neighbors, and their vote."""
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal([3.0, 3.2], [1.05, 0.95], (22, 2)),
                   rng.normal([6.2, 5.6], [1.05, 0.95], (22, 2))])
    y = np.repeat([0, 1], 22)
    q = np.array([4.75, 4.45])
    k = 5
    d = np.linalg.norm(X - q, axis=1)
    r_k = np.sort(d)[k - 1]
    keep = (d <= r_k) | (d > r_k + 0.55)      # clear a thin ring outside the circle so its edge reads cleanly
    X, y, d = X[keep], y[keep], d[keep]
    nn = np.argsort(d)[:k]
    votes = np.bincount(y[nn], minlength=2)
    winner = int(np.argmax(votes))

    fig = S.figure(9.6, 5.4)
    S.header(fig, "KNN: let the nearest neighbors vote",
             "To classify a new point, find its k = 5 closest training points. The majority label wins.")
    ax = fig.add_axes([0.065, 0.12, 0.56, 0.68])
    ax.add_patch(Circle(q, r_k + 0.25, fc=S.INK, alpha=0.05, ec="none", zorder=1))
    ax.add_patch(Circle(q, r_k + 0.25, fc="none", ec=S.MUTED, ls=(0, (4, 3)), lw=1.2, zorder=1))
    for i in nn:
        ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color=S.INK_2, lw=1.2, alpha=0.9, zorder=2)
    for cls in (0, 1):
        S.scatter(ax, X[y == cls, 0], X[y == cls, 1], cls)
    ring(ax, X[nn, 0], X[nn, 1])
    new_point(ax, *q)
    ax.annotate("new point:\nwhich class?", xy=q, xytext=(q[0] - 2.75, q[1] + 1.2), color=S.INK_2,
                fontsize=10.5, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkB=10))
    fit_equal(ax, 4.7, (1.25, 7.65))
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(handles=legend_handles(["Class 0", "Class 1"],
                                     [new_point_handle("New point"), ring_handle("Its 5 nearest neighbors")]),
              loc="lower right", fontsize=9.5, borderpad=0.2, handletextpad=0.4)

    # The tally: one ballot per neighbor
    panel = fig.add_axes([0.665, 0.2, 0.30, 0.52])
    S.clean_axes(panel)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    S.box(panel, (0, 0), 1, 1, "", radius=0.05)
    panel.text(0.08, 0.86, f"The vote  (k = {k})", fontsize=13, fontweight="bold", color=S.INK, va="center")
    order = [winner, 1 - winner]
    for row, cls in enumerate(order):
        yy = 0.64 - row * 0.18
        panel.text(0.08, yy, f"Class {cls}", fontsize=11.5, color=S.INK_2, va="center")
        xs = 0.40 + 0.085 * np.arange(votes[cls])
        S.scatter(panel, xs, np.full(xs.size, yy), cls, s=150)
        panel.text(0.92, yy, str(votes[cls]), fontsize=15, fontweight="bold", color=S.INK,
                   ha="right", va="center")
    panel.plot([0.08, 0.92], [0.33, 0.33], color=S.AXIS, lw=1, zorder=3)
    panel.text(0.08, 0.19, "Prediction:", fontsize=11.5, color=S.INK_2, va="center")
    panel.text(0.49, 0.19, f"Class {winner}", fontsize=13, fontweight="bold", color=S.INK, va="center")
    S.scatter(panel, [0.86], [0.19], winner, s=170)
    S.save(fig, OUT / "knn-vote.png")
    return votes


def decision_regions(ax, model, X, y, xlim, ylim, n=400):
    xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[S.BLUE, S.ORANGE], alpha=S.WASH, zorder=0)
    ax.contour(xx, yy, Z, levels=[0.5], colors=[S.INK_2], linewidths=1.0, zorder=1)


def choosing_k():
    X, y = make_moons(n_samples=300, noise=0.32, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=1 / 3, random_state=0, stratify=y)
    panels = [(1, False, "Overfits: chases every noisy point"),
              (15, True, "Just right: follows the real shape"),
              (150, False, "Underfits: too blunt to bend")]
    xlim, ylim = (-1.9, 2.9), (-1.45, 1.95)

    fig = S.figure(10.5, 5.0)
    S.header(fig, "Choosing k: from jumpy to blunt",
             "The same noisy data, three values of k. The shaded color is what the model predicts at each spot.")
    fig.legend(handles=legend_handles(["Class 0", "Class 1"]), loc="upper right",
               bbox_to_anchor=(0.975, 0.975), ncol=2, fontsize=10)
    for j, (k, good, verdict) in enumerate(panels):
        ax = fig.add_axes([0.035 + j * 0.327, 0.2, 0.29, 0.56])
        model = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        decision_regions(ax, model, Xtr, ytr, xlim, ylim)
        for cls in (0, 1):
            S.scatter(ax, Xtr[ytr == cls, 0], Xtr[ytr == cls, 1], cls, s=26)
        S.clean_axes(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"k = {k}", pad=8)
        tr, te = model.score(Xtr, ytr), model.score(Xte, yte)
        ax.text(0.0, -0.06, ("✓ " if good else "✗ ") + verdict, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, fontweight="bold", color=S.GOOD if good else S.CRITICAL)
        ax.text(0.0, -0.17, f"train accuracy {tr:.0%}   ·   test accuracy {te:.0%}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color=S.INK_2)
    S.save(fig, OUT / "choosing-k.png")


def fit_equal(ax, xcenter, ylim):
    """Equal aspect with no distortion: widen x so the limits match the panel's shape."""
    box = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    half = (ylim[1] - ylim[0]) * (box.width * fw) / (box.height * fh) / 2
    ax.set_xlim(xcenter - half, xcenter + half)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def sub_title(ax, text, color=S.INK_2, weight="normal"):
    """A second, smaller line under a panel title."""
    ax.text(0, 1.03, text, transform=ax.transAxes, fontsize=10.5, fontweight=weight, color=color, va="bottom")


def feature_scaling():
    Xtr, Xte, ytr, yte = customers()
    a = CUSTOMER_A
    k = 5
    raw = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    d_raw, nn_raw = (v[0] for v in raw.kneighbors([a]))
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(Xtr)
    az = scaler.transform([a])[0]
    scaled = KNeighborsClassifier(n_neighbors=k).fit(Z, ytr)
    d_z, nn_z = (v[0] for v in scaled.kneighbors([az]))

    fig = S.figure(10.5, 6.4)
    S.header(fig, "Scaling changes who counts as a neighbor",
             "Customer A is 25. Who are A's 5 nearest neighbors, before and after scaling age and income?")
    fig.legend(handles=legend_handles(["Didn't buy (0)", "Bought glasses (1)"],
                                      [new_point_handle("Customer A (25, $50k)"),
                                       ring_handle("A's 5 nearest neighbors")]),
               loc="upper left", bbox_to_anchor=(0.03, 0.845), ncol=4, fontsize=10, columnspacing=1.6)

    def panel(rect, title, P, p, nn, verdict_ok, xlabel, ylabel):
        ax = fig.add_axes(rect)
        for cls in (0, 1):
            S.scatter(ax, P[ytr == cls, 0], P[ytr == cls, 1], cls, s=24, alpha=0.95)
        for i in nn:
            ax.plot([p[0], P[i, 0]], [p[1], P[i, 1]], color=S.INK_2, lw=1.1, zorder=4)
        ring(ax, P[nn, 0], P[nn, 1], s=88)
        new_point(ax, *p, s=210)
        ages = Xtr[nn, 0]
        votes = np.bincount(ytr[nn], minlength=2)
        pred = "buys" if votes[1] > votes[0] else "won't buy"
        ax.set_title(title, pad=26)
        sub_title(ax, ("✓ " if verdict_ok else "✗ ") +
                  f"Neighbors aged {ages.min()} to {ages.max()}. Vote {votes[1]}–{votes[0]}: \"{pred}\"",
                  color=S.GOOD if verdict_ok else S.CRITICAL, weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    # Before: raw units. Distance only sees income, so "near A" is a thin stripe of similar incomes.
    P = np.column_stack([Xtr[:, 0], Xtr[:, 1] / 1000])
    p = np.array([a[0], a[1] / 1000])
    ax = panel([0.075, 0.125, 0.4, 0.56], "Before scaling: income decides everything", P, p, nn_raw,
               False, "Age (years)", "Income ($1,000s)")
    band = d_raw[-1] / 1000 + 0.35
    ax.axhspan(p[1] - band, p[1] + band, color=S.INK_2, alpha=0.18, lw=0, zorder=0)
    ax.text(82.5, p[1] + 3.2, "everyone in this\ngray stripe counts\nas \"close\" to A,\nat any age",
            color=S.INK_2, fontsize=9.5, ha="center", va="bottom", linespacing=1.25)
    ax.set_xlim(15, 95)
    ax.set_ylim(15, 125)

    # After: both features in "standard deviations". A circle is a fair neighborhood again.
    ax = panel([0.575, 0.125, 0.4, 0.56], "After scaling: age counts too", Z, az, nn_z,
               True, "Age (standardized)", "Income (standardized)")
    ax.add_patch(Circle(az, d_z[-1] + 0.07, fc=S.INK, alpha=0.07, ec=S.MUTED, lw=1.1, ls=(0, (3, 2)),
                        zorder=1))
    fit_equal(ax, 0.0, (-2.05, 1.95))
    S.save(fig, OUT / "feature-scaling.png")
    return Xtr[nn_raw], Xtr[nn_z]


def svm_street():
    X, y = make_blobs(n_samples=40, centers=[[2.2, 2.4], [5.6, 5.0]], cluster_std=0.75, random_state=4)
    svc = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = svc.coef_[0], svc.intercept_[0]
    ylim, xc = (-0.3, 7.5), 4.0
    xs = np.array([-10.0, 20.0])            # long enough to cross the whole panel

    def line(ww, bb, level=0.0, x=xs):        # y on the line ww . x + bb = level
        return (level - bb - ww[0] * x) / ww[1]

    fig = S.figure(10.5, 5.6)
    S.header(fig, "SVM: find the widest street between the classes",
             "Out of all the lines that separate two classes, SVM picks the one with the most room on both sides.")
    fig.legend(handles=legend_handles(["Class 0", "Class 1"], [ring_handle("Support vectors")]),
               loc="upper right", bbox_to_anchor=(0.975, 0.975), ncol=3, fontsize=10)

    # Left: any line that separates. Each candidate hugs some points.
    ax = fig.add_axes([0.06, 0.1, 0.41, 0.6])
    theta0 = np.arctan2(w[1], w[0])
    for dtheta, frac in [(np.deg2rad(24), 0.1), (np.deg2rad(-26), 0.88), (np.deg2rad(4), 0.93)]:
        u = np.array([np.cos(theta0 + dtheta), np.sin(theta0 + dtheta)])
        proj = X @ u
        lo, hi = proj[y == 0].max(), proj[y == 1].min()
        ax.plot(xs, line(u, -(lo + frac * (hi - lo))), color=S.INK_2, lw=1.3, alpha=0.9, zorder=2)
    for cls in (0, 1):
        S.scatter(ax, X[y == cls, 0], X[y == cls, 1], cls)
    ax.set_title("Many lines separate the classes", pad=26)
    sub_title(ax, "...but each of these passes close to some point.")
    fit_equal(ax, xc, ylim)

    # Right: the maximum-margin street
    ax = fig.add_axes([0.555, 0.1, 0.41, 0.6])
    ax.fill_between(xs, line(w, b, -1), line(w, b, 1), color=S.INK_2, alpha=0.1, lw=0, zorder=0)
    for level in (-1, 1):
        ax.plot(xs, line(w, b, level), color=S.MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.plot(xs, line(w, b), color=S.INK, lw=1.8, zorder=2)
    for cls in (0, 1):
        S.scatter(ax, X[y == cls, 0], X[y == cls, 1], cls)
    sv = svc.support_vectors_
    ring(ax, sv[:, 0], sv[:, 1], s=330)
    # A double-headed arrow straight across the street, and a label written along it
    n = w / np.linalg.norm(w)
    half = 1 / np.linalg.norm(w)
    mid = np.array([4.95, line(w, b, 0, 4.95)])
    ax.add_patch(FancyArrowPatch(mid - half * n, mid + half * n, arrowstyle="<|-|>", mutation_scale=12,
                                 color=S.INK, lw=1.3, zorder=4))
    angle = np.degrees(np.arctan2(-w[0], w[1]))
    tx = 2.2
    ax.text(tx, line(w, b, -0.5, tx), "the widest possible street", rotation=angle, rotation_mode="anchor",
            ha="center", va="center", fontsize=10, color=S.INK_2, zorder=3)
    ax.set_title("SVM picks the widest street", pad=26)
    sub_title(ax, "The ringed points on its edges are the support vectors.")
    fit_equal(ax, xc, ylim)
    S.save(fig, OUT / "svm-street.png")
    return len(sv)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def kernel_trick_gif():
    X, y = make_circles(n_samples=170, factor=0.42, noise=0.055, random_state=2)
    r2 = (X ** 2).sum(axis=1)                 # the new feature: z = x1^2 + x2^2
    cut = 0.5 * (r2[y == 1].max() + r2[y == 0].min())
    theta = np.linspace(0, 2 * np.pi, 120)

    # (name, frames, subtitle): each phase sets lift t, camera elevation, plane alpha, floor circle
    phases = [
        ("flat", 14, "In 2-D, no straight line can separate the inner ring from the outer ring."),
        ("tilt", 16, "Let's look at the same points from the side..."),
        ("lift", 24, "...and add one new feature, z = x₁² + x₂²: each point rises by its distance from the center."),
        ("plane", 14, "Now a flat plane slides right between the two classes."),
        ("hold", 8, "Now a flat plane slides right between the two classes."),
        ("back", 24, "Back in 2-D, that flat cut becomes a circle: a curved boundary."),
    ]
    frames = []
    for name, count, sub in phases:
        for i in range(count):
            f = ease((i + 1) / count)
            t, elev, azim, plane, circle = 0.0, 89.9, -90.0, 0.0, 0.0
            if name == "tilt":
                elev, azim = 89.9 - 71 * f, -90 + 30 * f
            elif name == "lift":
                t, elev, azim = f, 18.9, -60 + 8 * f
            elif name in ("plane", "hold"):
                t, elev, azim = 1.0, 18.9, -52 + (8 * f if name == "plane" else 8)
                plane = f if name == "plane" else 1.0
            elif name == "back":
                t, elev, azim = 1 - f, 18.9 + 71 * f, -44 - 46 * f
                plane, circle = 1 - f, f

            fig = S.figure(9, 5.4)
            S.header(fig, "The kernel trick: lift the data, then cut it flat", sub)
            ax = fig.add_axes([0.08, 0.02, 0.84, 0.84], projection="3d", computed_zorder=False)
            ax.set_proj_type("ortho")
            ax.set_facecolor(S.SURFACE)
            ax.set_axis_off()
            ax.view_init(elev=elev, azim=azim)
            # a faint floor grid for depth
            g = np.linspace(-1.25, 1.25, 6)
            for v in g:
                ax.plot([v, v], [g[0], g[-1]], [0, 0], color=S.AXIS, lw=0.7, zorder=0)
                ax.plot([g[0], g[-1]], [v, v], [0, 0], color=S.AXIS, lw=0.7, zorder=0)
            z = t * r2
            S.scatter(ax, X[y == 1, 0], X[y == 1, 1], 1, zs=z[y == 1], depthshade=False, s=34,
                      label="Class 1 (inner ring)")
            if plane > 0:
                pp = np.linspace(-1.3, 1.3, 2)
                PX, PY = np.meshgrid(pp, pp)
                ax.plot_surface(PX, PY, np.full_like(PX, cut * max(t, 1e-3)), color=S.INK_2,
                                alpha=0.22 * plane, shade=False, zorder=2)
            S.scatter(ax, X[y == 0, 0], X[y == 0, 1], 0, zs=z[y == 0], depthshade=False, s=34,
                      label="Class 0 (outer ring)")
            if circle > 0:
                rc = np.sqrt(cut)
                ax.plot(rc * np.cos(theta), rc * np.sin(theta), np.zeros_like(theta), color=S.INK,
                        lw=2.0, alpha=circle, zorder=4)
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
            ax.set_zlim(0, 1.45)
            ax.set_box_aspect((1, 1, 0.62), zoom=1.32)
            handles = legend_handles(["Class 0 (outer ring)", "Class 1 (inner ring)"])
            fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.03), fontsize=10)
            fig.text(0.985, 0.018, S.WATERMARK, fontsize=7.5, color=S.MUTED, ha="right", va="bottom")
            frames.append(S.fig_to_image(fig))
            S.plt.close(fig)
    return S.save_gif(frames, OUT / "kernel-trick.gif", fps=12)


if __name__ == "__main__":
    print("hero votes (class 0, class 1):", knn_vote())
    choosing_k()
    raw_nn, scaled_nn = feature_scaling()
    print("A's raw neighbors:\n", raw_nn, "\nA's scaled neighbors:\n", scaled_nn)
    print("support vectors in the street figure:", svm_street())
    kernel_trick_gif()
    print("lesson 05 diagrams ->", OUT)
