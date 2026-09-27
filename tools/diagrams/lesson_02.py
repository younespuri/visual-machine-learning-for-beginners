"""Diagrams for lesson 02 - Linear regression."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S  # noqa: E402

OUT = S.lesson_dir("02-linear-regression")

# The same houses the lesson's code uses
AREA = np.array([50, 60, 70, 85, 100, 120, 140])
PRICE = np.array([540, 560, 780, 790, 1040, 1080, 1340])


def best_fit_line():
    w, b = np.polyfit(AREA, PRICE, 1)
    fig = S.figure(9, 5.2)
    S.header(fig, "Linear regression draws the best straight line",
             "Each thin line is one error: the gap between the real price and the line's guess.")
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.66])
    xs = np.linspace(40, 150, 2)
    for x, y in zip(AREA, PRICE):
        ax.plot([x, x], [y, w * x + b], color=S.INK_2, lw=1.4, alpha=0.85, zorder=2)
    ax.plot(xs, w * xs + b, color=S.ORANGE, label=f"Model:  price = {w:.2f} × area + {b:.0f}", zorder=2)
    S.scatter(ax, AREA, PRICE, 0, label="Houses (the data)")
    i = int(np.argmax(np.abs(PRICE - (w * AREA + b))))
    x0, y0 = AREA[i], PRICE[i]
    ax.annotate("error = actual − predicted", xy=(x0, (y0 + w * x0 + b) / 2), xytext=(x0 + 12, y0 - 260),
                color=S.INK_2, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1))
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price ($1,000s)")
    ax.set_xlim(40, 150)
    ax.set_ylim(350, 1450)
    ax.legend(loc="upper left")
    S.save(fig, OUT / "best-fit-line.png")


def squared_errors():
    rng = np.random.default_rng(7)
    x = np.array([1.2, 2.3, 3.1, 4.4, 5.2, 6.3, 7.4, 8.6])
    y = 0.75 * x + 1.4 + rng.normal(0, 0.55, x.size)
    w_best, b_best = np.polyfit(x, y, 1)
    lines = [("A poor line", 0.15, 4.6), ("The best line", w_best, b_best)]

    fig = S.figure(10, 5.4)
    S.header(fig, "Why square the errors?",
             "Each error becomes a real square. MSE is the average area: the best line has the smallest squares.")
    for i, (name, w, b) in enumerate(lines):
        ax = fig.add_axes([0.06 + i * 0.49, 0.09, 0.42, 0.66])
        pred = w * x + b
        for xi, yi, pi in zip(x, y, pred):
            side = yi - pi
            direction = 1 if xi < 5 else -1   # grow squares inward so none are clipped
            ax.add_patch(Rectangle((xi, pi), direction * abs(side), side, fc=S.ORANGE,
                                   alpha=S.WASH + 0.07, ec=S.ORANGE, lw=0.9, zorder=1))
        xs = np.array([0, 10])
        ax.plot(xs, w * xs + b, color=S.ORANGE, zorder=2)
        S.scatter(ax, x, y, 0)
        mse = np.mean((y - pred) ** 2)
        ax.set_title(f"{name}:  MSE = {mse:.2f}")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))
    S.save(fig, OUT / "squared-errors.png")


def gradient_descent_gif(steps=45):
    rng = np.random.default_rng(3)
    x = np.linspace(1, 9, 14)
    y = 0.8 * x + 1.2 + rng.normal(0, 0.6, x.size)
    mx = x.mean()

    # Gradient descent on MSE. Centring x keeps the two parameters from fighting,
    # so the line settles smoothly; the plotted line is converted back to y = w*x + b.
    w, c, lr = -0.6, 8.0, 0.05
    history = []
    for _ in range(steps):
        err = w * (x - mx) + c - y
        history.append((w, c - w * mx, np.mean(err ** 2)))
        w -= lr * 2 * np.mean(err * (x - mx))
        c -= lr * 2 * np.mean(err)

    losses = [h[2] for h in history]
    frames = []
    for k, (w_k, b_k, loss_k) in enumerate(history):
        fig = S.figure(9, 5)
        S.header(fig, "Gradient descent: the model learns by shrinking its error",
                 f"Step {k + 1:>2}:   w = {w_k:5.2f}    b = {b_k:5.2f}    MSE = {loss_k:6.2f}")
        ax = fig.add_axes([0.07, 0.12, 0.47, 0.64])
        xs = np.array([0, 10])
        for j in range(max(0, k - 6), k):          # a fading trail of recent lines
            wj, bj, _ = history[j]
            ax.plot(xs, wj * xs + bj, color=S.ORANGE, alpha=0.08 + 0.04 * (j - k + 6), lw=1)
        ax.plot(xs, w_k * xs + b_k, color=S.ORANGE, label="the model's line")
        S.scatter(ax, x, y, 0, label="data", s=55)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("The line moves toward the data")
        ax.legend(loc="lower right", fontsize=9)

        ax2 = fig.add_axes([0.62, 0.12, 0.34, 0.64])
        ax2.plot(range(1, k + 2), losses[:k + 1], color=S.ORANGE)
        ax2.scatter([k + 1], [loss_k], s=55, color=S.ORANGE, edgecolors=S.SURFACE,
                    linewidths=S.RING_W, zorder=3)
        ax2.set_xlim(0, steps + 1)
        ax2.set_ylim(0, losses[0] * 1.08)
        ax2.set_xlabel("step")
        ax2.set_ylabel("MSE (error)")
        ax2.set_title("...so the error goes down")
        fig.text(0.985, 0.018, S.WATERMARK, fontsize=7.5, color=S.MUTED, ha="right", va="bottom")
        frames.append(S.fig_to_image(fig))
        S.plt.close(fig)
    return S.save_gif(frames, OUT / "gradient-descent.gif", fps=12)


if __name__ == "__main__":
    best_fit_line()
    squared_errors()
    gradient_descent_gif()
    print("lesson 02 diagrams ->", OUT)
