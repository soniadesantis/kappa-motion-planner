"""Three connected corridors and a magnified corner, using planner geometry."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, ConnectionPatch
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import get_corner_point_and_intersecting_edges
from tangent_convention_journal import STYLE
from example_one_corridor import arrow, text, edge_label, robot

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
ROBOT_RADIUS = 0.20
CORNER_COLOR = 'red'
FONT_SCALE = 1.18


def build_geometry():
    corridors = [CorridorWorld(1.1, 4.0, [0, 0], np.pi / 6),
                 CorridorWorld(1.4, 4.0, [1.73, 2.0], np.pi / 2),
                 CorridorWorld(1.1, 5.0, [3.3, 3.1], -np.pi / 6)]
    shrunk = [corridor.shrink(ROBOT_RADIUS) for corridor in corridors]
    turns = [int(np.sign(a.unit_vector[0] * b.unit_vector[1] -
                         a.unit_vector[1] * b.unit_vector[0]))
             for a, b in zip(corridors, corridors[1:])]
    corners = [get_corner_point_and_intersecting_edges(a, b, turn)
               for a, b, turn in zip(corridors, corridors[1:], turns)]
    if any(point is None for point, edges in corners):
        raise ValueError('The configured corridor sequence has no valid connecting corner.')
    return corridors, shrunk, turns, corners


def boundary(ax, edge, center, extent=2, **kwargs):
    normal = edge[:2]
    foot = center - (normal @ center + edge[2]) * normal / (normal @ normal)
    tangent = np.array([-normal[1], normal[0]])
    ax.plot(*np.array([foot - extent * tangent, foot + extent * tangent]).T, **kwargs)


def create_figure():
    corridors, shrunk, turns, corners = build_geometry()
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10.5, 8))
        ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
        for i, (corridor, inset) in enumerate(zip(corridors, shrunk), 1):
            ax.add_patch(Polygon(corridor.corners, fill=False, edgecolor='black', lw=1.5))
            ax.add_patch(Polygon(inset.corners, fill=False, edgecolor='black', lw=1.4, linestyle='--'))
            center = np.array(corridor.center)
            ax.plot(*center, 'ko', ms=5)
            arrow(ax, center, center + 0.48 * np.array(corridor.unit_vector))
        for position, label in [((-1.67, 0.1), r'$\mathcal{C}_1$'),
                                ((0.74, 3.30), r'$\mathcal{C}_2$'),
                                ((3.57, 3.93), r'$\mathcal{C}_3$')]:
            text(ax, *position, label, size=27)
        first = corridors[0]
        start = np.array(first.center) - 0.9 * np.array(first.unit_vector)
        start += (first.width / 2 - ROBOT_RADIUS) * first.outward_normals[first.RGT]
        # The footprint is tangent to the original boundary, centered on its inset.
        ax.add_patch(Circle(start, ROBOT_RADIUS, facecolor='0.72',
                            edgecolor='black', lw=2, zorder=5))
        arrow(ax, start, start + 0.48 * np.array([0.5, np.sqrt(3) / 2]), lw=1.8, zorder=8)
        label_positions = [(0.20, 1.63), (2.98, 2.12)]
        for index, ((point, edges), label_position) in enumerate(zip(corners, label_positions), 1):
            ax.plot(*point, 'o', color=CORNER_COLOR, ms=7, zorder=8)
            edge_label(ax, rf'$\mathbf{{p}}_{index}^{{\mathrm{{corn}}}}$',
                       label_position, point, CORNER_COLOR, 0.2)
            ax.texts[-1].set_color(CORNER_COLOR)
        text(ax, 0.02, 2.10, r'$\tau_1=+1$', size=25)
        text(ax, 3.18, 1.63, r'$\tau_2=-1$', size=25)
        point, edges = corners[1]
        box_half = 0.31
        ax.add_patch(Rectangle(point - box_half, 2 * box_half, 2 * box_half,
                               fill=False, edgecolor='0.65', lw=1.0, zorder=9))
        ax.set(xlim=(-2.25, 5.95), ylim=(-1.68, 4.95), aspect='equal')
        ax.axis('off')
        detail = fig.add_axes((0.64, 0.06, 0.33, 0.33))
        detail.set_aspect('equal')
        detail.set_xlim(point[0] - 0.56, point[0] + 0.56)
        detail.set_ylim(point[1] - 0.52, point[1] + 0.52)
        detail.set_xticks([])
        detail.set_yticks([])
        for spine in detail.spines.values():
            spine.set_color('0.65')
            spine.set_linewidth(0.9)
        for corridor, inset, edge_index in zip(corridors[1:], shrunk[1:], edges):
            boundary(detail, corridor.W[:, edge_index], point, color='black', lw=1.6)
            boundary(detail, inset.W[:, edge_index], point, color='black', lw=1.5, linestyle='--')
        # Region between the inset-edge intersection and the radius-r corner disk.
        normals = np.array([corridors[i + 1].W[:2, edge] for i, edge in enumerate(edges)])
        constants = np.array([shrunk[i + 1].W[2, edge] for i, edge in enumerate(edges)])
        inset_corner = np.linalg.solve(normals, -constants)
        contacts = point - ROBOT_RADIUS * normals
        angles = np.arctan2((contacts - point)[:, 1], (contacts - point)[:, 0])
        delta = (angles[1] - angles[0] + np.pi) % (2 * np.pi) - np.pi
        values = np.linspace(angles[0], angles[0] + delta, 80)
        arc = point + ROBOT_RADIUS * np.array([np.cos(values), np.sin(values)]).T
        detail.add_patch(Polygon(np.vstack([inset_corner, arc]), color='#FFB38A',
                                 alpha=0.6, lw=0, zorder=1))
        detail.add_patch(Circle(point, ROBOT_RADIUS, fill=False, color=CORNER_COLOR, lw=2, zorder=5))
        detail.plot(*point, 'o', color=CORNER_COLOR, ms=7, zorder=6)
        arrow(detail, point, point + ROBOT_RADIUS * np.array([-0.65, -0.76]),
              style='<->', lw=1.1, zorder=7)
        edge_label(detail, r'$\mathbf{p}_2^{\mathrm{corn}}$',
                   point + [0.25, -0.34], point, CORNER_COLOR, 0.2)
        detail.texts[-1].set_color(CORNER_COLOR)
        text(detail, *(point + [-0.40, -0.24]), r'$\bar{e}_{2,2}$', size=21)
        text(detail, *(point + [0.43, 0.15]), r'$\bar{e}_{2,3}$', size=21)
        edge_label(detail, r'$\mathcal{W}_2^{\mathrm{corn}}$',
                   point + [0.29, 0.36], inset_corner + [0.06, -0.04], '#E89866', -0.25)
        detail.patches[-1].shrinkB = 38
        connector = ConnectionPatch(point + [box_half, 0], (0.52, 1),
                                    coordsA=ax.transData, coordsB=detail.transAxes,
                                    color='0.65', lw=0.9, zorder=0)
        fig.add_artist(connector)
        for panel in (ax, detail):
            for label in panel.texts:
                label.set_fontsize(label.get_fontsize() * FONT_SCALE)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            path = OUTPUT_DIRECTORY / f'example_three_corridors_recreated.{extension}'
            figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {path}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
