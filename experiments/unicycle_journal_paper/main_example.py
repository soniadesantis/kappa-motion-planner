"""Journal main example, migrated from plots_for_figures/main_example.py.

Uses the original geometry and the planner's standing-assumptions mode.
All annotations and the magnified corner are generated in Python.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, ConnectionPatch, Arc
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle
from kappa_planner.trajectory import CurvilinearArcUnicycle, TurnOnTheSpot
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector, get_corner_point
from kappa_planner.helpers.poses import pose_from_shrunken_corridor_relative_frame
from tangent_convention_journal import STYLE
from example_one_corridor import arrow, text, edge_label

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
COLORS = ('#0000FF', '#BE00BE', '#008000')
FONT_SCALE = 1.15
THIRD_CORRIDOR_HEAD_EXTENSION = 0.4


def extend_head(corridor, distance):
    """Extend forward while preserving the tail and both side-wall lines."""
    corridor.update(height=corridor.height + distance,
                    center=np.asarray(corridor.center) +
                    0.5 * distance * np.asarray(corridor.unit_vector))


def build_problem():
    vehicle = Unicycle(model='Rosbot circular')
    vehicle.update(v_max=0.6, omega_max=1)
    corridors = [CorridorWorld(1.3, 3, [0, 0], 0)]
    for width, length, angle in [(1, 3, np.pi/2), (1, 3, -np.pi/6),
                                  (1.2, 2.3, -np.pi/2 - 0.2)]:
        tail = np.array(corridors[-1].head)
        head = tail + length * np.array([np.cos(angle), np.sin(angle)])
        corridors.append(get_corridor_from_vector(tail, head, width, add_height=0.7))
    # Resolve the original start pose before changing its corridor's length.
    start_pose = pose_from_shrunken_corridor_relative_frame(
        [0.8, -0.75, np.pi + 0.7], corridors[0].shrink(vehicle.width / 2))
    # Extend after constructing the chain, so the other corridors stay in place.
    right_wall = np.asarray(corridors[1].get_edge_segment(CorridorWorld.RGT)[0])
    right_normal = np.asarray(corridors[1].outward_normals[CorridorWorld.RGT])
    extension = np.dot(right_wall - corridors[0].head, right_normal) / np.dot(
        corridors[0].unit_vector, right_normal)
    extend_head(corridors[0], extension)
    # Extend the second corridor backwards to the first corridor's right wall.
    right_wall = np.asarray(corridors[0].get_edge_segment(CorridorWorld.RGT)[0])
    right_normal = np.asarray(corridors[0].outward_normals[CorridorWorld.RGT])
    backward = -np.asarray(corridors[1].unit_vector)
    tail_extension = np.dot(right_wall - corridors[1].tail, right_normal) / np.dot(
        backward, right_normal)
    corridors[1].update(height=corridors[1].height + tail_extension,
                        center=np.asarray(corridors[1].center) +
                        0.5 * tail_extension * backward)
    extend_head(corridors[2], THIRD_CORRIDOR_HEAD_EXTENSION)
    planner = MotionPlanner(vehicle, corridors,
                            start_pose=start_pose,
                            relative_end_pose=[-0.2, 0.8, 0], assumptions='standing')
    trajectory = planner.compute_trajectory_analytical()
    return planner, trajectory


def pose(ax, xy, heading, radius):
    ax.add_patch(Circle(xy, radius, fill=False, edgecolor='black', lw=1.5, zorder=8))
    arrow(ax, xy, np.array(xy) + (radius + 0.13) * np.array([np.cos(heading), np.sin(heading)]),
          lw=1.1, zorder=9)


def path(ax, primitive, color, linewidth=4.0):
    if isinstance(primitive, TurnOnTheSpot):
        return
    if isinstance(primitive, CurvilinearArcUnicycle):
        start = np.arctan2(primitive.y0 - primitive.yc, primitive.x0 - primitive.xc)
        end = np.arctan2(primitive.yf - primitive.yc, primitive.xf - primitive.xc)
        turn = primitive.turn_direction
        sweep = turn * ((turn * (end - start)) % (2 * np.pi))
        values = np.linspace(start, start + sweep, 180)
        ax.plot(primitive.xc + primitive.radius * np.cos(values),
                primitive.yc + primitive.radius * np.sin(values), color=color, lw=linewidth, zorder=5)
    else:
        ax.plot([primitive.x0, primitive.xf], [primitive.y0, primitive.yf],
                color=color, lw=linewidth, zorder=5)


def create_figure():
    planner, trajectory = build_problem()
    corridors = planner.corridor_list
    radius = planner.vehicle.width / 2
    arcs = [p for p in trajectory if isinstance(p, CurvilinearArcUnicycle)]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11.5, 8.4))
        ax = fig.add_axes((0.01, 0.09, 0.98, 0.90))
        for corridor in corridors:
            ax.add_patch(Polygon(corridor.corners, fill=False, edgecolor='black', lw=1.2))
            ax.add_patch(Polygon(corridor.shrink(radius).corners, fill=False,
                                 edgecolor='black', lw=1.1, linestyle='--'))
        circle_names = ['0', '1', '2', '3', 'f']
        offsets = [(0.35, 0.10), (-0.20, 0.37), (0.15, -0.32), (-0.30, -0.26), (0.08, 0.43)]
        for arc, name, offset in zip(arcs, circle_names, offsets):
            color = 'red' if name in ('1', '2', '3') else 'black'
            center = np.array([arc.xc, arc.yc])
            ax.add_patch(Circle(center, arc.radius, fill=False, color=color, lw=0.9, linestyle='--'))
            ax.plot(*center, 'ko', ms=2)
            text(ax, *(center + offset), rf'$\mathcal{{O}}_{name}$', size=25, color=color)
        for i, primitive in enumerate(trajectory):
            path(ax, primitive, COLORS[0 if i < 4 else 1 if i < 7 else 2])
        poses = [planner.start_pose] + [[p.xf, p.yf, p.thetaf] for p in trajectory]
        seen = set()
        for x, y, heading in poses:
            key = tuple(np.round([x, y, heading], 7))
            if key not in seen:
                pose(ax, (x, y), heading, radius)
                seen.add(key)
        corner = get_corner_point(corridors[0], corridors[1], 1)
        center = np.array([arcs[1].xc, arcs[1].yc])
        ax.add_patch(Rectangle(center - [0.68, 0.47], 1.20, 1.15,
                               fill=False, edgecolor='0.65', lw=0.9))
        ax.set(xlim=(-2.4, 5.25), ylim=(-1.08, 4.05), aspect='equal')
        ax.axis('off')
        detail = fig.add_axes((0.025, 0.48, 0.25, 0.31))
        detail.set(xlim=(corner[0] - 1.02, corner[0] + 0.24),
                   ylim=(corner[1] - 0.16, corner[1] + 1.04), aspect='equal')
        detail.set_xticks([])
        detail.set_yticks([])
        for spine in detail.spines.values():
            spine.set_color('0.65')
            spine.set_linewidth(0.9)
        for corridor in corridors[:2]:
            detail.add_patch(Polygon(corridor.corners, fill=False, edgecolor='black', lw=1.3))
            detail.add_patch(Polygon(corridor.shrink(radius).corners, fill=False,
                                     edgecolor='black', lw=1.2, linestyle='--'))
        detail.add_patch(Circle(center, arcs[1].radius, fill=False, color='red', lw=1.0, linestyle='--'))
        path(detail, arcs[1], COLORS[0], linewidth=4.2)
        detail.plot(*center, 'ko', ms=6, zorder=12)
        detail.plot(*corner, 'o', color='red', ms=5, zorder=10)
        vector = center - corner
        arrow(detail, corner, center, style='<->', lw=1.8)
        detail.annotate(r'$R-r$', xy=corner + 0.42 * vector,
                        xytext=corner + [-0.70, 0.22], fontsize=25,
                        ha='center', va='center', zorder=12,
                        arrowprops=dict(arrowstyle='<|-', color='black', lw=1.2,
                                        linestyle=':', connectionstyle='arc3,rad=-0.15',
                                        shrinkA=7, shrinkB=0))
        detail.annotate(r'$\mathbf{p}_1^o$', xy=center,
                        xytext=center + [-0.39, 0.24], fontsize=25,
                        ha='center', va='center', zorder=12,
                        arrowprops=dict(arrowstyle='<|-', color='black', lw=1.2,
                                        linestyle=':', connectionstyle='arc3,rad=-0.15',
                                        shrinkA=7, shrinkB=3))
        detail.annotate(r'$\mathbf{p}_1^{\mathrm{corn}}$', xy=corner,
                        xytext=corner + [-0.59, -0.34], fontsize=25, color='red',
                        ha='center', va='center', annotation_clip=False, zorder=12,
                        arrowprops=dict(arrowstyle='<|-', color='red', lw=1.3,
                                        linestyle=':', connectionstyle='arc3,rad=0.30',
                                        shrinkA=7, shrinkB=3, clip_on=False))
        text(detail, *(center + [-0.10, 0.43]), r'$\mathcal{O}_1$', size=25, color='red')
        xi = np.arctan2(vector[1], vector[0])
        detail.add_patch(Arc(corner, 0.46, 0.46, theta1=0, theta2=np.degrees(xi), color='0.5', lw=1.2))
        arrow(detail, corner + 0.23 * np.array([np.cos(xi - 0.15), np.sin(xi - 0.15)]),
              corner + 0.23 * np.array([np.cos(xi), np.sin(xi)]), color='0.5', lw=1.2)
        text(detail, *(corner + [-0.08, 0.30]), r'$\xi_1$', size=24)
        fig.add_artist(ConnectionPatch((1, 0.5), center + [-0.08, 0.68],
                       coordsA=detail.transAxes, coordsB=ax.transData, color='0.65', lw=0.9, zorder=0))
        # Build the notation from the computed primitives, including their actual turn signs.
        groups = [[], [], []]
        for index, primitive in enumerate(trajectory, 1):
            group = 0 if index <= 4 else 1 if index <= 7 else 2
            if isinstance(primitive, TurnOnTheSpot):
                name = 'T'
            elif isinstance(primitive, CurvilinearArcUnicycle):
                name = 'C'
            else:
                name = 'S'
            sign = '' if name == 'S' else ('+' if primitive.turn_direction > 0 else '-')
            groups[group].append(rf'{name}_{{{index}}}' + (rf'^{{{sign}}}' if sign else ''))
        for xpos, group, color in zip((0.38, 0.62, 0.85), groups, COLORS):
            fig.text(xpos, 0.035, '$' + r'\,'.join(group) + '$', color=color,
                     fontsize=23, ha='center', va='center')
        for panel in (ax, detail):
            for annotation in panel.texts:
                annotation.set_fontsize(annotation.get_fontsize() * FONT_SCALE)
        for annotation in fig.texts:
            annotation.set_fontsize(annotation.get_fontsize() * FONT_SCALE)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            path = OUTPUT_DIRECTORY / f'main_example_recreated.{extension}'
            figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {path}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
