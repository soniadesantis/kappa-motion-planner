"""Recreate the journal's single-corridor geometry illustration.

The source PDF is preserved. Exports go to the journal figures directory.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, FancyBboxPatch, Arc
import numpy as np

from tangent_convention_journal import STYLE

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
EDGE_COLORS = ('#66B84A', '#E89866', '#55BED8', '#C875C9')
GREY = '0.62'
RADIUS = 0.40


def arrow(ax, start, end, color='black', style='-|>', lw=1.3, **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                            color=color, lw=lw, shrinkA=0, shrinkB=0, **kwargs)
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, size=25, color='black', **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color,
                   ha=kwargs.pop('ha', 'center'), va=kwargs.pop('va', 'center'), **kwargs)


def robot(ax, center, radius=RADIUS):
    ax.add_patch(Circle(center, radius, facecolor='0.72', edgecolor='black', lw=2, zorder=5))
    ax.plot(*center, 'ko', ms=5, zorder=7)
    heading = 2.12
    end = np.array(center) + 2.05 * radius * np.array([np.cos(heading), np.sin(heading)])
    arrow(ax, center, end, lw=1.8, zorder=8)


def angular_arrow(ax, center, radius, end_angle):
    ax.add_patch(Arc(center, 2 * radius, 2 * radius, theta1=0,
                     theta2=np.degrees(end_angle), color=GREY, lw=1.4, zorder=6))
    start = np.asarray(center) + radius * np.array([np.cos(end_angle - 0.15),
                                                   np.sin(end_angle - 0.15)])
    end = np.asarray(center) + radius * np.array([np.cos(end_angle), np.sin(end_angle)])
    arrow(ax, start, end, color=GREY, lw=1.4, zorder=7)


def corridor(ax, x, colored=False):
    width, height, bottom = 2.8, 6.4, 0.6
    outer = np.array([[x, bottom + height], [x + width, bottom + height],
                      [x + width, bottom], [x, bottom]])
    inner = np.array([[x + RADIUS, bottom + height - RADIUS],
                      [x + width - RADIUS, bottom + height - RADIUS],
                      [x + width - RADIUS, bottom + RADIUS],
                      [x + RADIUS, bottom + RADIUS]])
    if colored:
        for vertices in (outer, inner):
            for i, color in enumerate(EDGE_COLORS):
                segment = vertices[[i, (i + 1) % 4]]
                ax.plot(*segment.T, color=color, lw=9, alpha=0.25,
                        solid_capstyle='round', zorder=0)
    ax.add_patch(Rectangle((x, bottom), width, height, fill=False, lw=1.5))
    ax.add_patch(Rectangle((x + RADIUS, bottom + RADIUS), width - 2 * RADIUS,
                           height - 2 * RADIUS, fill=False, lw=1.4, linestyle='--'))
    center = np.array([x + width / 2, bottom + height / 2])
    ax.plot(*center, 'ko', ms=5)
    arrow(ax, center, center + [0, 1.0])
    pose = np.array([x + width - RADIUS, 2.6])
    robot(ax, pose)
    return center, pose


def edge_label(ax, label, position, target, color, curvature=0.15):
    text(ax, *position, label)
    # Stop before the glyphs, keeping the coloured leader short.
    patch = FancyArrowPatch(target, position, arrowstyle='-|>',
                            connectionstyle=f'arc3,rad={curvature}',
                            color=color, lw=1.2, linestyle=':',
                            mutation_scale=11, shrinkA=3, shrinkB=18)
    ax.add_patch(patch)


def create_figure():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 6.6))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        c1, _ = corridor(ax, 2.0, colored=True)
        center, pose = corridor(ax, 6.7)
        # Edge ordering follows the uploaded reference: top, right, bottom, left.
        placements = [
            (r'$e_{1,1}$', (3.05, 7.50), (4.20, 7.00), 0),
            (r'$\bar{e}_{1,1}$', (3.12, 5.95), (3.95, 6.60), 0),
            (r'$e_{2,1}$', (5.45, 4.95), (4.80, 5.45), 1),
            (r'$\bar{e}_{2,1}$', (3.85, 5.02), (4.40, 5.60), 1),
            (r'$e_{3,1}$', (3.80, 0.12), (3.00, 0.60), 2),
            (r'$\bar{e}_{3,1}$', (3.08, 1.52), (3.90, 1.00), 2),
            (r'$e_{4,1}$', (1.32, 4.02), (2.00, 4.60), 3),
            (r'$\bar{e}_{4,1}$', (3.03, 3.00), (2.40, 3.48), 3),
        ]
        for label, position, target, index in placements:
            edge_label(ax, label, position, target, EDGE_COLORS[index])
        legend = ax.legend(handles=[
            Line2D([], [], color='black', lw=1.5, label=r'$\mathcal{C}_1$'),
            Line2D([], [], color='black', lw=1.5, linestyle='--', label=r'$\bar{\mathcal{C}}_1$')],
            loc='upper left', bbox_to_anchor=(0.0, 0.85), frameon=False,
            fontsize=27, handlelength=1.5)
        # Coordinate frame and positive angular direction.
        origin = np.array([0.25, 1.20])
        arrow(ax, origin, origin + [0.85, 0], lw=1.6)
        arrow(ax, origin, origin + [0, 0.95], lw=1.6)
        text(ax, 1.22, 1.08, r'$\mathbf{x}$')
        text(ax, 0.08, 2.22, r'$\mathbf{y}$')
        arrow(ax, (0.91, 1.55), (0.48, 1.94), connectionstyle='arc3,rad=0.65')
        text(ax, 0.71, 1.71, r'$+$', size=27)
        # Corridor dimensions, inset and orientation.
        arrow(ax, (6.7, 7.4), (9.5, 7.4), color=GREY, style='<->')
        text(ax, 8.1, 7.68, r'$w_1$')
        arrow(ax, (6.25, 0.6), (6.25, 7.0), color=GREY, style='<->')
        text(ax, 5.96, 3.8, r'$l_1$')
        arrow(ax, (7.35, 6.6), (7.35, 7.0), color=GREY, style='<->')
        text(ax, 7.62, 6.81, r'$r$')
        text(ax, center[0] - 0.30, center[1] + 1.02, r'$\mathbf{v}_1$')
        ax.plot([center[0], center[0] + 0.75], [center[1]] * 2, 'k--', lw=1)
        angular_arrow(ax, center, 0.45, np.pi / 2)
        text(ax, center[0] + 0.54, center[1] + 0.49, r'$\psi_1$')
        edge_label(ax, r'$(x_1^c,y_1^c)$', (7.85, 2.96), center, 'black', 0.2)
        text(ax, 7.75, 1.55, r'$(x,y)$')
        ax.add_patch(FancyArrowPatch((7.75, 1.55), pose, arrowstyle='<|-',
                     connectionstyle='arc3,rad=-0.2', color='black',
                     lw=1.2, linestyle=':', mutation_scale=11,
                     shrinkA=18, shrinkB=0, zorder=10))
        # Magnified footprint at the inset right boundary.
        small_box = (8.43, 1.72, 1.35, 1.80)
        large_box = (10.15, 2.05, 2.80, 4.05)
        for x, y, width, height in (small_box, large_box):
            ax.add_patch(FancyBboxPatch((x, y), width, height,
                         boxstyle='round,pad=0.03,rounding_size=0.25',
                         fill=False, edgecolor=GREY, lw=1.1, linestyle=':'))
        ax.plot([9.81, 10.12], [2.75, 3.83], ':', color=GREY, lw=1.2)
        text(ax, 11.55, 6.47, 'Unicycle', size=23)
        zoom_center = np.array([11.30, 3.95])
        zoom_radius = 0.72
        ax.plot([11.30, 11.30], [2.06, 6.10], 'k--', lw=1.5)
        ax.plot([12.02, 12.02], [2.06, 6.10], 'k-', lw=1.5)
        robot(ax, zoom_center, zoom_radius)
        ax.plot([11.30, 12.60], [3.95, 3.95], 'k--', lw=1, zorder=7)
        angular_arrow(ax, zoom_center, 1.15, 2.12)
        text(ax, 12.32, 4.99, r'$\theta$')
        endpoint = zoom_center + zoom_radius * np.array([-0.65, -0.76])
        arrow(ax, zoom_center, endpoint, style='<->', lw=1.3, zorder=9)
        text(ax, 10.96, 3.86, r'$r$', size=27, zorder=10)
        ax.set(xlim=(-0.18, 13.15), ylim=(-0.12, 7.97), aspect='equal')
        ax.axis('off')
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    fig = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            output = OUTPUT_DIRECTORY / f'example_one_corridor_recreated.{extension}'
            fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
