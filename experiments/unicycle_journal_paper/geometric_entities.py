"""Exact point-to-circle tangents for concentric circles of radii R and 2R."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyArrowPatch
import numpy as np

from tangent_convention_journal import STYLE

RADIUS = 1.0
POINT = np.array([0., 0.])
CENTER = np.array([0., 4.0])
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
# Left and right tangent colors for each circle.
COLORS = {1: ('#9EBB22', '#229D39'), 2: ('#FF812C', '#DA3036')}


def tangency_points(point, center, radius):
    offset = point - center
    squared_distance = offset @ offset
    if squared_distance <= radius**2:
        raise ValueError('The point must lie strictly outside both circles.')
    perpendicular = np.array([-offset[1], offset[0]])
    foot = center + radius**2 / squared_distance * offset
    shift = radius * np.sqrt(squared_distance - radius**2) / squared_distance * perpendicular
    return foot - shift, foot + shift


def label(ax, position, symbol, color='black', size=28):
    return ax.text(*position, symbol, color=color, fontsize=size,
                   ha='center', va='center', zorder=10)


def leader(ax, position, target, symbol):
    ax.annotate(symbol, xy=target, xytext=position, fontsize=29,
                ha='center', va='center', zorder=12,
                arrowprops=dict(arrowstyle='<|-', color='black', lw=1.2,
                                linestyle=':', connectionstyle='arc3,rad=0.30',
                                shrinkA=5, shrinkB=0))


def angle_arc(ax, start, end, radius, symbol, color, label_radius=None, label_fraction=0.5):
    ax.add_patch(Arc(POINT, 2 * radius, 2 * radius,
                     theta1=min(start, end), theta2=max(start, end),
                     color=color, lw=1.5, zorder=6))
    middle = np.radians(start + label_fraction * (end - start))
    distance = radius + 0.18 if label_radius is None else label_radius
    label(ax, POINT + distance * np.array([np.cos(middle), np.sin(middle)]), symbol, color)


def create_figure():
    direction = CENTER - POINT
    alpha = np.degrees(np.arctan2(direction[1], direction[0]))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 9.5))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
        for multiplier in (2, 1):
            ax.add_patch(Circle(CENTER, multiplier * RADIUS, fill=False,
                                edgecolor='0.2', lw=1.5, zorder=2))
        ax.plot([POINT[0], CENTER[0]], [POINT[1], CENTER[1]], '--', color='0.55', lw=1)
        ax.plot([-2.0, 2.0], [POINT[1], POINT[1]], '--', color='0.55', lw=1)
        label(ax, CENTER + [-0.51, 0.66], r'$\mathcal{O}_1$', size=31)
        label(ax, CENTER + [-1.30, 1.30], r"$\mathcal{O}'_1$", size=31)
        for multiplier in (1, 2):
            radius = multiplier * RADIUS
            contacts = tangency_points(POINT, CENTER, radius)
            for side, (contact, color) in enumerate(zip(contacts, COLORS[multiplier])):
                ax.plot(*np.vstack([POINT, contact]).T, color=color, lw=1.4, zorder=3)
                ax.plot(*np.vstack([CENTER, contact]).T, color='0.55', lw=1.0, zorder=1)
                inward = (CENTER - contact) / radius
                along = (POINT - contact) / np.linalg.norm(POINT - contact)
                size = 0.12
                square = np.array([contact + size * inward,
                                   contact + size * (inward + along),
                                   contact + size * along])
                ax.plot(*square.T, color='#163A47', lw=1.6, zorder=7)
                if side == 0:
                    position = CENTER + (0.72 if multiplier == 1 else 0.85) * (contact - CENTER)
                    label(ax, position + [-0.02, 0.17], r'$R$' if multiplier == 1 else r'$2R$')
                tangent_angle = np.degrees(np.arctan2(*(contact - POINT)[::-1]))
                arc_radius = (1.34 if multiplier == 1 else 1.78) + 0.045 * side
                angle_arc(ax, alpha, tangent_angle, arc_radius,
                          r'$\beta$' if multiplier == 1 else r"$\beta'$", color,
                          label_radius=arc_radius + (0.25 if multiplier == 1 else 0.19),
                          label_fraction=0.5 if multiplier == 1 else 0.82)
        angle_arc(ax, 0, alpha, 0.73, r'$\alpha_0$', '0.5', label_radius=0.96)
        end_angle = np.radians(alpha)
        arrow_start = POINT + 0.73 * np.array([np.cos(end_angle - 0.15),
                                              np.sin(end_angle - 0.15)])
        arrow_end = POINT + 0.73 * np.array([np.cos(end_angle), np.sin(end_angle)])
        ax.add_patch(FancyArrowPatch(arrow_start, arrow_end, arrowstyle='-|>',
                                    mutation_scale=12, color='0.5', lw=1.5,
                                    shrinkA=0, shrinkB=0, zorder=7))
        for point in (POINT, CENTER):
            ax.plot(*point, 'ko', ms=4.5, zorder=11)
        leader(ax, POINT + [-1.05, 0.52], POINT, r'$(x_0,y_0)$')
        leader(ax, CENTER + [0.43, 0.43], CENTER, r'$\mathbf{p}_1^o$')
        ax.set(xlim=(-2.13, 2.13), ylim=(-0.15, 6.10), aspect='equal')
        ax.axis('off')
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            path = OUTPUT_DIRECTORY / f'geometric_entities_recreated.{extension}'
            figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {path}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
