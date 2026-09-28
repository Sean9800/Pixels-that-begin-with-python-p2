import sys

import numpy as np
import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

TEXT = ""
FPS = 20
SAVE = "--save" in sys.argv
rng = np.random.default_rng(11)
T_OUTLINE = (0.6, 7.0)
T_FILL = (5.5, 11.5)
T_TEXT = (11.0, 13.5)
T_BEAT = 13.0
TOTAL = 18.0


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def prog(t, span):
    if span[1] <= span[0]:
        return 0.0
    return ease((t - span[0]) / (span[1] - span[0]))


def heart(t):
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y


def build_heart_pixels(t, width=72, height=72):
    x = np.linspace(-18.0, 18.0, width)
    y = np.linspace(-18.0, 18.0, height)
    xx, yy = np.meshgrid(x, y)

    # Classic heart equation used as a pixel mask.
    heart_mask = ((xx ** 2 + yy ** 2 - 1) ** 3 - xx ** 2 * yy ** 3 <= 0)

    pulse = 1.0 + 0.15 * np.sin(2.0 * np.pi * (t / T_BEAT))
    scale = 1.4 + 0.3 * prog(t, T_OUTLINE)
    heart_shape = heart_mask & (np.sqrt((xx / (pulse * scale)) ** 2 + (yy / (pulse * scale)) ** 2) <= 1.18)

    rows, cols = np.nonzero(heart_shape)
    if rows.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    x_coords = xx[rows, cols]
    y_coords = yy[rows, cols]
    brightness = np.linspace(0.15, 1.0, len(x_coords))
    brightness += 0.25 * np.sin((x_coords + y_coords) * 0.9 + t)
    brightness = np.clip(brightness, 0.0, 1.0)
    return x_coords, y_coords, brightness


def animate_heart():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("black")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = LinearSegmentedColormap.from_list(
        "heart_palette",
        [
            (0.0, "#18050e"),
            (0.25, "#5e1035"),
            (0.5, "#ff2d75"),
            (0.75, "#ff9ac8"),
            (1.0, "#fff6ff"),
        ],
    )

    text_alpha = 0.0
    if TEXT:
        text_alpha = 1.0

    def update(frame):
        ax.clear()
        ax.set_facecolor("black")
        ax.set_xlim(-18, 18)
        ax.set_ylim(-18, 18)
        ax.set_aspect("equal")
        ax.axis("off")

        time = frame / FPS
        x_coords, y_coords, brightness = build_heart_pixels(time)

        if x_coords.size > 0:
            colors = cmap(brightness)
            sizes = 55 + 80 * brightness
            ax.scatter(x_coords, y_coords, s=sizes, c=colors, marker="s", edgecolors="none")

        outline = prog(time, T_OUTLINE)
        outline_scale = 1.0 + 0.25 * outline
        theta = np.linspace(0, 2 * np.pi, 600)
        x_curve, y_curve = heart(theta)
        x_curve *= outline_scale
        y_curve *= outline_scale
        ax.plot(x_curve, y_curve, color="#ffd2e6", linewidth=1.5, alpha=0.6)

        if TEXT:
            t_text = prog(time, T_TEXT)
            txt = ax.text(
                0,
                -11.5,
                TEXT,
                ha="center",
                va="center",
                fontsize=18,
                color=(1.0, 0.9, 0.96, 0.2 + 0.8 * t_text),
                family="sans-serif",
            )
            txt.set_path_effects([])

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, TOTAL, int(FPS * TOTAL), endpoint=False),
        interval=1000 / FPS,
        blit=False,
    )

    if SAVE:
        animation.save("pixel_heart.gif", writer=PillowWriter(fps=FPS))
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    animate_heart()
