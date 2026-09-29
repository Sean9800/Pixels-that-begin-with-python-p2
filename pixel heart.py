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

    pulse = 1.0 + 0.15 * np.sin(2.0 * np.pi * (t / T_BEAT))
    scale = 1.4 + 0.3 * prog(t, T_OUTLINE)
    normalized_x = xx / (14.0 * pulse * scale)
    normalized_y = yy / (14.0 * pulse * scale)

    # Evaluate the classic equation in its unit-sized coordinate system.
    heart_shape = (
        (normalized_x ** 2 + normalized_y ** 2 - 1) ** 3
        - normalized_x ** 2 * normalized_y ** 3
        <= 0
    )

    rows, cols = np.nonzero(heart_shape)
    if rows.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    x_coords = xx[rows, cols]
    y_coords = yy[rows, cols]
    y_range = max(y_coords.max() - y_coords.min(), 1.0)
    vertical_gradient = (y_coords - y_coords.min()) / y_range
    highlight = 1.0 - np.clip(np.abs(normalized_x[rows, cols]), 0.0, 1.0)
    brightness = 0.2 + 0.55 * vertical_gradient + 0.2 * highlight
    brightness += 0.08 * np.sin((x_coords + y_coords) * 0.9 + t)
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
            ax.scatter(x_coords, y_coords, s=45, c=colors, marker="s", edgecolors="none")

        border_x = np.linspace(-18.0, 18.0, 240)
        border_y = np.linspace(-18.0, 18.0, 240)
        border_xx, border_yy = np.meshgrid(border_x, border_y)
        pulse = 1.0 + 0.15 * np.sin(2.0 * np.pi * (time / T_BEAT))
        scale = 1.4 + 0.3 * prog(time, T_OUTLINE)
        normalized_x = border_xx / (14.0 * pulse * scale)
        normalized_y = border_yy / (14.0 * pulse * scale)
        heart_boundary = (
            (normalized_x ** 2 + normalized_y ** 2 - 1) ** 3
            - normalized_x ** 2 * normalized_y ** 3
        )
        ax.contour(
            border_xx,
            border_yy,
            heart_boundary,
            levels=[0],
            colors="black",
            linewidths=2.0,
        )

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
