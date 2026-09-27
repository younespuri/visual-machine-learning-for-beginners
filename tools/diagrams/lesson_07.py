"""Diagrams for lesson 07 - Clustering and PCA."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("07-clustering-and-pca")

# The same unlabeled blobs the lesson's code uses (Step 2 and the bonus)
X_BLOBS, _ = make_blobs(n_samples=150, centers=3, random_state=38)

# Three deliberately bad starting centroids, all crowded along the top edge
BAD_START = np.array([[4.0, 9.0], [6.0, 8.5], [10.5, 9.5]])

# The three digits shown in the PCA plot (the lesson's "See it" code plots the same ones)
DIGITS_SHOWN = (0, 4, 7)


def kmeans_rounds(X, start):
    """Run plain K-Means by hand and keep every round: (labels, old centroids, new centroids)."""
    centroids, rounds, previous = start.copy(), [], None
    while True:
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = dist.argmin(axis=1)
        if previous is not None and np.array_equal(labels, previous):
            return rounds
        moved = np.array([X[labels == j].mean(axis=0) for j in range(len(centroids))])
        rounds.append((labels, centroids, moved))
        centroids, previous = moved, labels


def _loop_panel(ax, active):
    """The two-step loop, with the step being shown on the left highlighted."""
    S.clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [("1 · Assign", "each point joins its\nnearest centroid", 0.70),
             ("2 · Move", "each centroid moves to\nthe mean of its points", 0.36)]
    for name, text, y in steps:
        on = active == name[0]
        S.box(ax, (0.02, y), 0.80, 0.25, "", fc=S.PANEL if on else S.SURFACE,
              ec=S.INK if on else S.AXIS, lw=1.6 if on else 1.0, radius=0.04)
        ax.text(0.42, y + 0.175, name, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color=S.INK if on else S.MUTED, transform=ax.transAxes, zorder=3)
        ax.text(0.42, y + 0.075, text, ha="center", va="center", fontsize=9.5, linespacing=1.25,
                color=S.INK_2 if on else S.MUTED, transform=ax.transAxes, zorder=3)
    done = active == "done"
    S.arrow(ax, (0.42, 0.695), (0.42, 0.615), color=S.MUTED)
    S.arrow(ax, (0.83, 0.485), (0.83, 0.825), color=S.INK if done else S.MUTED,
            connectionstyle="arc3,rad=0.55", clip_on=False)
    ax.text(0.42, 0.31, "repeat until no point\nchanges its cluster", ha="center", va="top",
            fontsize=9.5, linespacing=1.25, color=S.INK if done else S.MUTED,
            fontweight="bold" if done else "normal", transform=ax.transAxes)
    handles = [Line2D([], [], ls="", marker=m, ms=7, mfc=c, mec=S.SURFACE, label=f"cluster {j + 1}")
               for j, (c, m) in enumerate(zip(S.SERIES, S.MARKERS))]
    handles += [Line2D([], [], ls="", marker="o", ms=12, mfc=S.MUTED, mec=S.INK, mew=2, label="centroid"),
                Line2D([], [], color=S.MUTED, lw=1.3, ls=(0, (3, 2)), marker="o", ms=5, mfc=S.SURFACE,
                       mec=S.MUTED, label="its path")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, -0.04), ncol=2, fontsize=9,
              handletextpad=0.4, columnspacing=0.8, labelspacing=0.9)


def _kmeans_figure(X, labels, centroids, paths, subtitle, active, spokes=False):
    fig = S.figure(9, 5)
    S.header(fig, "K-Means finds groups by repeating two simple steps", subtitle)
    ax = fig.add_axes([0.04, 0.07, 0.62, 0.74])
    S.clean_axes(ax)
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=48, c=S.MUTED, edgecolors=S.SURFACE,
                   linewidths=S.RING_W, zorder=3)
    else:
        if spokes:
            segs = [[p, centroids[j]] for p, j in zip(X, labels)]
            ax.add_collection(LineCollection(segs, colors=[S.SERIES[j] for j in labels],
                                             linewidths=0.8, alpha=0.3, zorder=2))
        for j in range(len(centroids)):
            S.scatter(ax, X[labels == j, 0], X[labels == j, 1], j, s=48)
    for j, path in enumerate(paths):
        path = np.asarray(path)
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], color=S.SERIES[j], lw=1.3, ls=(0, (3, 2)), zorder=4)
            ax.scatter(path[:-1, 0], path[:-1, 1], s=46, marker=S.MARKERS[j], facecolors=S.SURFACE,
                       edgecolors=S.SERIES[j], linewidths=1.3, zorder=4)
    for j, c in enumerate(centroids):
        ax.scatter(*c, s=620, marker=S.MARKERS[j], c=S.SURFACE, zorder=5)
        ax.scatter(*c, s=330, marker=S.MARKERS[j], c=S.SERIES[j], edgecolors=S.INK,
                   linewidths=2.2, zorder=6)
    ax.set_xlim(-6, 12.5)
    ax.set_ylim(-1.2, 11)
    ax.set_aspect("equal")
    _loop_panel(fig.add_axes([0.70, 0.08, 0.27, 0.72]), active)
    return fig


def _kmeans_frame(*args, **kwargs):
    fig = _kmeans_figure(*args, **kwargs)
    fig.text(0.985, 0.018, S.WATERMARK, fontsize=7.5, color=S.MUTED, ha="right", va="bottom")
    image = S.fig_to_image(fig)
    S.plt.close(fig)
    return image


def kmeans_frames(X=X_BLOBS, start=BAD_START, slide=4):
    """Frames for a 4 fps GIF. Stills are repeated so each step stays up long enough to read;
    the first two rounds get the most time, later rounds are small nudges and go faster."""
    rounds = kmeans_rounds(X, start)
    paths = [[c] for c in start]
    frames = [_kmeans_frame(X, None, start, paths,
                            "Start · 3 centroids are dropped in bad spots on purpose", None)] * 8
    for r, (labels, old, new) in enumerate(rounds, 1):
        slow = r <= 2
        frames += [_kmeans_frame(X, labels, old, paths,
                                 f"Round {r} · Assign: each point joins its nearest centroid",
                                 "1", spokes=True)] * (6 if slow else 4)
        for t in np.linspace(0, 1, slide + 1)[1:]:
            now = old + (new - old) * (1 - (1 - t) ** 2)          # ease out
            trail = [p + [c] for p, c in zip(paths, now)]
            frames.append(_kmeans_frame(X, labels, now, trail,
                                        f"Round {r} · Move: each centroid moves to the mean of its points",
                                        "2"))
        paths = [p + [c] for p, c in zip(paths, new)]
        frames += [frames[-1]] * (4 if slow else 2)
    labels, _, final = rounds[-1]
    frames.append(_kmeans_frame(X, labels, final, paths, _done_text(rounds), "done"))
    return frames, rounds


def _done_text(rounds):
    return f"Done after {len(rounds)} rounds · no point changed its cluster, so K-Means stops"


def kmeans_gif():
    frames, rounds = kmeans_frames()
    S.save_gif(frames, OUT / "kmeans-in-action.gif", fps=4, hold_last=12)
    # The last frame again as a crisp static PNG, for places where a GIF doesn't fit
    labels, _, final = rounds[-1]
    paths = [[BAD_START[j]] + [r[2][j] for r in rounds] for j in range(len(BAD_START))]
    S.save(_kmeans_figure(X_BLOBS, labels, final, paths, _done_text(rounds), "done"),
           OUT / "kmeans-in-action.png")


def _ring(ax, x, y, s=300):
    """A white ring that marks the chosen point on a chart."""
    ax.scatter([x], [y], s=s, facecolors="none", edgecolors=S.INK, linewidths=1.8, zorder=4)


def elbow():
    ks = np.arange(1, 9)
    fits = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_BLOBS) for k in ks]
    inertia = np.array([f.inertia_ for f in fits])
    sil = np.array([silhouette_score(X_BLOBS, f.labels_) for f in fits[1:]])

    fig = S.figure(10, 5.4)
    S.header(fig, "How many clusters? Look for the elbow",
             "Inertia always drops as k grows. Pick the k where the big drops stop: here, k = 3.")
    ax = fig.add_axes([0.085, 0.13, 0.50, 0.62])
    ax.plot(ks, inertia, color=S.BLUE, zorder=2)
    S.scatter(ax, ks, inertia, 0)
    _ring(ax, 3, inertia[2])
    ax.annotate("the elbow: k = 3", xy=(3, inertia[2]), xytext=(3.9, 1650), color=S.INK,
                fontsize=11.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkB=9))
    ax.text(1.6, 3500, "big drops: each new\ncluster helps a lot", color=S.INK_2, fontsize=10,
            va="top", linespacing=1.35)
    ax.text(5.2, 900, "small drops: extra\nclusters barely help", color=S.INK_2, fontsize=10,
            va="top", linespacing=1.35)
    ax.set_title("Elbow method: inertia for each k")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("inertia (lower = tighter clusters)")
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 8.4)
    ax.set_ylim(0, inertia[0] * 1.1)

    ax2 = fig.add_axes([0.685, 0.13, 0.285, 0.62])
    ax2.plot(ks[1:], sil, color=S.BLUE, zorder=2)
    S.scatter(ax2, ks[1:], sil, 0)
    best = int(ks[1:][sil.argmax()])
    _ring(ax2, best, sil.max())
    ax2.annotate(f"best: k = {best}", xy=(best, sil.max()), xytext=(best + 1.3, sil.max() + 0.015),
                 color=S.INK, fontsize=11.5, fontweight="bold", va="center",
                 arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1, shrinkB=9))
    ax2.set_title("Second opinion: silhouette")
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("silhouette (higher = better)")
    ax2.set_xticks(ks[1:])
    ax2.set_xlim(1.6, 8.4)
    ax2.set_ylim(0, 0.85)
    S.save(fig, OUT / "elbow-method.png")


def _voronoi_wash(ax, km, order, xlim, ylim):
    """Fill each K-Means region with its cluster hue: every spot takes the nearest centroid's color."""
    xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    zz = order[km.predict(np.c_[xx.ravel(), yy.ravel()])].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=S.SERIES[:2], alpha=S.WASH, zorder=0)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[S.INK_2], linewidths=1.1, linestyles="--", zorder=1)


def _disk(rng, n, center, radius):
    """n points spread evenly over a disk, so the group has a crisp edge."""
    r = radius * np.sqrt(rng.random(n))
    angle = 2 * np.pi * rng.random(n)
    return np.c_[center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)]


def kmeans_limits():
    moons, _ = make_moons(n_samples=200, noise=0.06, random_state=0)
    rng = np.random.default_rng(1)
    uneven = np.vstack([_disk(rng, 220, (0.0, 0.0), 3.0),     # one big group
                        _disk(rng, 30, (4.3, 0.0), 0.5)])     # one small, tight group
    # Both panels get the same width:height ratio, so their boxes line up
    panels = [(moons, "Curved shapes: each moon gets cut", (-1.6, 2.6), (-1.34, 1.84)),
              (uneven, "Uneven sizes: the big group gets split", (-3.8, 5.4), (-3.48, 3.48))]

    fig = S.figure(10, 5.2)
    S.header(fig, "Where K-Means goes wrong",
             "Its border is always a straight line halfway between two centroids, "
             "so it expects round, similar-sized groups.")
    for i, (X, title, xlim, ylim) in enumerate(panels):
        ax = fig.add_axes([0.035 + i * 0.49, 0.08, 0.44, 0.64])
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
        order = np.argsort(np.argsort(km.cluster_centers_[:, 0]))      # left cluster = BLUE, right = ORANGE
        labels = order[km.labels_]
        _voronoi_wash(ax, km, order, xlim, ylim)
        for j in range(2):
            S.scatter(ax, X[labels == j, 0], X[labels == j, 1], j, s=30,
                      label=f"cluster {j + 1}" if i == 0 else None)
        for c, j in zip(km.cluster_centers_, order):
            ax.scatter(*c, s=520, marker=S.MARKERS[j], c=S.SURFACE, zorder=5)
            ax.scatter(*c, s=260, marker=S.MARKERS[j], c=S.SERIES[j], edgecolors=S.INK,
                       linewidths=2, zorder=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        S.clean_axes(ax)
        ax.set_title(title, loc="left")
        if i == 0:
            ax.legend(loc="lower left", fontsize=9.5, handletextpad=0.3, borderaxespad=0.2)
    S.save(fig, OUT / "kmeans-limits.png")


def pca_directions():
    rng = np.random.default_rng(4)
    pts = rng.multivariate_normal([0, 0], [[3.0, 1.6], [1.6, 1.8]], size=36)
    pca = PCA(n_components=2).fit(pts)
    mean, (pc1, pc2) = pca.mean_, pca.components_
    pc1 = pc1 if pc1[0] > 0 else -pc1          # point the arrows up and to the right
    pc2 = pc2 if pc2[1] > 0 else -pc2
    sd1, sd2 = np.sqrt(pca.explained_variance_)
    r1, r2 = pca.explained_variance_ratio_
    along = (pts - mean) @ pc1                   # each point's position on PC1
    shadow = mean + np.outer(along, pc1)

    fig = S.figure(10, 5.6)
    S.header(fig, "PCA finds the directions where the data spreads out most",
             "PC1 follows the longest spread and PC2 sits at a right angle. "
             "Keep only PC1 and each point becomes one number.")

    # Left: the data, the two new axes, and every point's shadow on PC1
    ax = fig.add_axes([0.06, 0.10, 0.55, 0.68])
    t = np.array([-12, 12])
    ax.plot(mean[0] + t * pc1[0], mean[1] + t * pc1[1], color=S.ORANGE, lw=1, ls=(0, (4, 3)), alpha=0.6,
            zorder=1)
    for p, q in zip(pts, shadow):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=S.INK_2, lw=0.9, alpha=0.75, zorder=2)
    S.scatter(ax, pts[:, 0], pts[:, 1], 0, label="data point")
    ax.scatter(shadow[:, 0], shadow[:, 1], s=22, c=S.INK, edgecolors=S.SURFACE, linewidths=0.8, zorder=4,
               label="its shadow on PC1")
    for v, sd in ((pc1, sd1), (pc2, sd2)):
        ax.annotate("", xy=mean + 2 * sd * v, xytext=mean, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=2.8, mutation_scale=20,
                                    shrinkA=0, shrinkB=0))
    end1, end2 = mean + 2 * sd1 * pc1, mean + 2 * sd2 * pc2
    ax.text(*(end1 + pc1 * 0.25), "PC1", color=S.INK, fontsize=12, fontweight="bold", ha="left",
            va="top")
    ax.text(*(end2 + pc2 * 0.12), "PC2", color=S.INK, fontsize=12, fontweight="bold", ha="right",
            va="bottom")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_aspect("equal")
    ax.set_xlim(-5.2, 5.2)                       # same width:height ratio as the axes box
    ax.set_ylim(-3.6, 3.6)
    ax.legend(loc="lower right", fontsize=9.5, handletextpad=0.3)

    # Top right: what is left after dropping PC2
    ax1 = fig.add_axes([0.685, 0.53, 0.28, 0.13])
    ax1.axhline(0, color=S.ORANGE, lw=1, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax1.scatter(along, np.zeros_like(along), s=22, c=S.INK, edgecolors=S.SURFACE, linewidths=0.8, zorder=3)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.grid(False)
    ax1.set_ylim(-1, 1)
    ax1.set_xlim(-4.6, 4.6)
    ax1.set_xlabel("position along PC1")
    ax1.set_title("After PCA: 1 number per point", fontsize=11.5)

    # Bottom right: how much of the spread each direction carries
    ax2 = fig.add_axes([0.685, 0.12, 0.28, 0.20])
    ax2.barh([1, 0], [r1, r2], height=0.55, color=S.ORANGE, zorder=2)
    for y, r in zip([1, 0], [r1, r2]):
        ax2.text(r + 0.03, y, f"{r:.0%}", va="center", color=S.INK, fontsize=11, fontweight="bold")
    ax2.set_yticks([1, 0], ["PC1", "PC2"])
    ax2.set_xlim(0, 1.18)
    ax2.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Share of the spread each one keeps", fontsize=11.5)
    S.save(fig, OUT / "pca-directions.png")


def digits_pca():
    digits = load_digits()
    pca = PCA(n_components=2)
    z = pca.fit_transform(digits.data)
    kept = pca.explained_variance_ratio_.sum()
    fig = S.figure(10, 5.6)
    S.header(fig, "PCA squeezes 64 pixels into 2 numbers",
             f"Each point is one handwritten digit. The 2 components keep only {kept:.0%} of the variance, "
             "yet each digit gets its own spot.")

    # Left: one image is just 64 numbers
    sample = digits.images[0]                     # the dataset's first image, a clear 0
    axi = fig.add_axes([0.035, 0.30, 0.19, 0.34])
    axi.imshow(sample, cmap=S.SEQ, vmin=0, vmax=16)
    for k in np.arange(-0.5, 8, 1):
        axi.axhline(k, color=S.SURFACE, lw=0.8)
        axi.axvline(k, color=S.SURFACE, lw=0.8)
    S.clean_axes(axi)
    fig.text(0.13, 0.25, "one 8 × 8 image\n= 64 numbers", ha="center", va="top", color=S.INK_2,
             fontsize=10.5, linespacing=1.3)
    fig.text(0.268, 0.50, "PCA", ha="center", va="bottom", color=S.INK, fontsize=11, fontweight="bold")
    fig.add_artist(FancyArrowPatch((0.235, 0.47), (0.305, 0.47), arrowstyle="-|>", mutation_scale=16,
                                   color=S.MUTED, lw=1.8, transform=fig.transFigure))

    # Right: every image of 0, 4 and 7 as a point in 2-D
    ax = fig.add_axes([0.37, 0.12, 0.60, 0.66])
    for j, d in enumerate(DIGITS_SHOWN):
        m = digits.target == d
        S.scatter(ax, z[m, 0], z[m, 1], j, s=30, alpha=0.9, label=f"digit {d}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=10, handletextpad=0.3)
    S.save(fig, OUT / "digits-pca.png")


if __name__ == "__main__":
    kmeans_gif()
    elbow()
    kmeans_limits()
    pca_directions()
    digits_pca()
    print("lesson 07 diagrams ->", OUT)
