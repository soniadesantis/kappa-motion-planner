"""Construction geometry for bicycle CSC and C^bCSC candidates in a 2x2 grid.

Uses horizontally separated poses and the radius of trajectory_constructions.py. Signs in
panel titles denote heading rotation; the backward arc has opposite steering.
Run with --no-show to regenerate the PDF and PNG without opening a window.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Polygon, Wedge
from matplotlib.lines import Line2D

from kappa_planner.trajectory import BackwardArc, CurvilinearArcUnicycle, LinearSegmentUnicycle
from trajectory_constructions import (
    GREY, RADIUS, STYLE, path,
)

from kappa_planner.geometry import Pose, Point, IntermediateCircle
from kappa_planner.vehicle import Bicycle
from kappa_planner.helpers.pose_to_circle_bicycle import (
    compute_traj_to_circle_free_space_bicycle, compute_backward_arc_optimal,
    compute_two_maneuvers_bicycle,
)
from kappa_planner.helpers.geometry_operations import compute_angular_difference_with_turn_direction

START_POSITION = np.array([0., 0.])
END_POSITION = np.array([5., 0.])
END_HEADING_DEGREES = 135.
PANEL_CONFIGURATIONS = [
    ('CSC', 1, 1, -45), ('CSC', -1, 1, 45),
    ('CbCSC', 1, 1, 180), ('CbCSC', -1, 1, 135),
]


def build_panel(config):
    family, tau0, taut, initial_heading = config
    vehicle = Bicycle(model='Bicycle circular')
    vehicle.update(v_max=RADIUS, v_min=-RADIUS,
                   delta_max=np.arctan(vehicle.wheelbase/RADIUS))
    start = Pose(Point(*START_POSITION), np.radians(initial_heading))
    end = Pose(Point(*END_POSITION), np.radians(END_HEADING_DEGREES))
    center = END_POSITION+RADIUS*np.array([np.cos(end.theta+taut*np.pi/2),
                                          np.sin(end.theta+taut*np.pi/2)])
    target = IntermediateCircle(center=Point(*center), radius=RADIUS,
                                turn_direction=taut, corner_point=Point(*END_POSITION))
    if family == 'CbCSC':
        # Construct the requested family explicitly, including when the
        # automatic selection rule would prefer a forward-only candidate.
        backward = compute_backward_arc_optimal(start, tau0, taut, target, vehicle)
        arc, segment = compute_two_maneuvers_bicycle(
            [backward.xf, backward.yf, backward.thetaf], vehicle, target, taut,
            t0=backward.tf, tau1=tau0)
        primitives = [backward, arc, segment]
    else:
        primitives = compute_traj_to_circle_free_space_bicycle(start, vehicle, target, tau0=tau0)
    assert isinstance(primitives[0], BackwardArc) == (family == 'CbCSC')
    contact = primitives[-1]
    sweep = compute_angular_difference_with_turn_direction(contact.thetaf, end.theta, taut)
    primitives.append(CurvilinearArcUnicycle(
        xc=center[0], yc=center[1], x0=contact.xf, y0=contact.yf,
        theta0=contact.thetaf, xf=end.x, yf=end.y, thetaf=contact.thetaf+sweep,
        radius=RADIUS, turn_direction=taut, v=vehicle.v_max,
        omega=taut*vehicle.omega_max, unicycle=vehicle, t0=contact.tf, samples_number=100))
    return primitives, primitives[-1].tf


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
BACKWARD_COLOR = '#A16B00'
AUXILIARY_COLOR = '0.65'
FONT_SIZE = 17
PATH_COLOR = 'tab:blue'
START_POSE_COLOR = 'tab:green'
END_POSE_COLOR = 'tab:red'
POSE_ARROW_LENGTH = 0.8
SECTOR_COLOR = '#CC79A7'
SECTOR_LABEL_COLOR = '#884D70'
SECTOR_ALPHA = 0.3
SECTOR_RADIUS_RATIO = 0.4


def heading(ax, point, angle, color=PATH_COLOR, length=POSE_ARROW_LENGTH):
    """Match the long, narrow pose arrows in the eight-panel unicycle figure."""
    point = np.asarray(point)
    ax.plot(*point, 'o', color=color, ms=4, zorder=100)
    end = point+length*np.array([np.cos(angle), np.sin(angle)])
    ax.add_patch(FancyArrowPatch(
        point, end, arrowstyle='-|>,head_length=0.6,head_width=0.2',
        mutation_scale=7, color=color, lw=2, joinstyle='miter',
        capstyle='butt', shrinkA=0, shrinkB=0, zorder=101))


def label(ax, point, text, offset=(5, 5), color='black', **kwargs):
    ax.annotate(text, xy=point, xytext=offset, textcoords='offset points',
                fontsize=FONT_SIZE, color=color, zorder=20, **kwargs)



def label_outside_circle(ax, point, center, text, color, gap=5):
    """Place a boundary-position label just beyond its outward radius."""
    outward = np.asarray(point)-np.asarray(center)
    outward /= np.linalg.norm(outward)
    label(ax, point, text, tuple(gap*outward), color=color,
          ha='left' if outward[0] > 1e-8 else 'right' if outward[0] < -1e-8 else 'center',
          va='bottom' if outward[1] > 1e-8 else 'top' if outward[1] < -1e-8 else 'center')


def line(ax, first, last, color=GREY, style='--', width=0.85):
    ax.plot(*np.array([first, last]).T, linestyle=style, color=color,
            lw=width, zorder=2)


def supporting_circle(ax, center, name, center_name, circle_offset, color=GREY):
    ax.add_patch(Circle(center, RADIUS, fill=False, color=color,
                        linestyle='--', lw=0.9, zorder=1))
    ax.plot(*center, 'o', color='black', ms=3, zorder=8)


def dimension(ax, first, last, text, offset=(0, 7), color=GREY):
    ax.add_patch(FancyArrowPatch(first, last, arrowstyle='<->',
                                mutation_scale=9, color=color, lw=0.8,
                                shrinkA=0, shrinkB=0, zorder=3))
    label(ax, (first+last)/2, text, offset, ha='center', color=color)


def angle(ax, vertex, start, end, text, radius=0.48, color=GREY):
    sweep = np.arctan2(np.sin(end-start), np.cos(end-start))
    low, high = sorted(np.degrees([start, start+sweep]))
    ax.add_patch(Arc(vertex, 2*radius, 2*radius, theta1=low,
                     theta2=high, color=color, lw=0.9, zorder=4))
    mid = start+sweep/2
    position = vertex+(radius+0.15)*np.array([np.cos(mid), np.sin(mid)])
    if text:
        label(ax, position, text, (0, 0), ha='center', va='center', color=color)


def arc_sector(ax, primitive, center_name, amplitude_name, square=False):
    """Shade the actual arc sweep; align right-angle squares with its radii."""
    center = np.array([primitive.xc, primitive.yc])
    start = np.arctan2(primitive.y0-primitive.yc, primitive.x0-primitive.xc)
    end = np.arctan2(primitive.yf-primitive.yc, primitive.xf-primitive.xc)
    turn = (-primitive.turn_direction if isinstance(primitive, BackwardArc)
            else primitive.turn_direction)
    sweep = turn*((turn*(end-start)) % (2*np.pi))
    label_angle = start+sweep/2+np.pi
    label(ax, center, center_name,
          (10*np.cos(label_angle), 10*np.sin(label_angle)),
          ha='center', va='center', color='black')
    amplitude_angle = start+sweep/2
    label_radius = (SECTOR_RADIUS_RATIO+0.2)*primitive.radius
    label_position = center+label_radius*np.array([np.cos(amplitude_angle), np.sin(amplitude_angle)])
    label(ax, label_position, amplitude_name, (0, 0), ha='center', va='center',
          color=SECTOR_LABEL_COLOR)
    radius = SECTOR_RADIUS_RATIO*primitive.radius
    style = dict(facecolor=SECTOR_COLOR, edgecolor=SECTOR_COLOR,
                 alpha=SECTOR_ALPHA, linewidth=0.9, zorder=3)
    if square:
        assert np.isclose(abs(sweep), np.pi/2)
        side = radius/np.sqrt(2)
        u = side*np.array([np.cos(start), np.sin(start)])
        v = side*np.array([np.cos(start+sweep), np.sin(start+sweep)])
        ax.add_patch(Polygon([center, center+u, center+u+v, center+v],
                             closed=True, **style))
    else:
        low, high = sorted(np.degrees([start, start+sweep]))
        ax.add_patch(Wedge(center, radius, theta1=low, theta2=high, **style))


def annotate_segment_length(ax, start, end, offset_ratio=0.45):
    """Use the eight-panel figure's offset dimension and extension lines."""
    delta = end-start
    length = np.linalg.norm(delta)
    if np.isclose(length, 0):
        return
    normal = np.array([delta[1], -delta[0]])/length
    offset = offset_ratio*RADIUS
    side = 1 if offset_ratio >= 0 else -1
    dimension_points = [point+offset*normal for point in (start, end)]
    for point in (start, end):
        extension = point+(offset+side*0.07*RADIUS)*normal
        ax.plot(*np.array([point, extension]).T, color=GREY, linestyle='--',
                linewidth=0.7, scalex=False, scaley=False, zorder=1)
    ax.add_patch(FancyArrowPatch(
        *dimension_points, arrowstyle='<->', shrinkA=0, shrinkB=0,
        mutation_scale=10, color='black', linewidth=0.9, zorder=2))
    label(ax, (dimension_points[0]+dimension_points[1])/2, r'$d$',
          (0, -5*side), color='black', ha='center',
          va='top' if side > 0 else 'bottom')


def draw_panel(ax, config, letter):
    family, tau0, taut, initial_heading = config
    trajectory, duration = build_panel(config)
    forward = [p for p in trajectory if isinstance(p, CurvilinearArcUnicycle)]
    segment = next(p for p in trajectory if isinstance(p, LinearSegmentUnicycle))
    o0 = np.array([forward[0].xc, forward[0].yc])
    ot = np.array([forward[-1].xc, forward[-1].yc])
    q0 = np.array([segment.x0, segment.y0])
    qt = np.array([segment.xf, segment.yf])
    direction = (qt-q0)/np.linalg.norm(qt-q0)
    supporting_circle(ax, o0, r'$\mathcal O_0$', r'$\mathbf o_0$',
                      (-0.60, 0.40) if tau0==taut else (0.60, -0.36))
    supporting_circle(ax, ot, r'$\mathcal O_1$', r'$\mathbf o_1$', (-0.62, -0.37))
    # Match the gray dotted radius guides in the eight-panel figure.
    # Each actual arc contributes its two endpoint poses; the reflected
    # circle is auxiliary geometry and receives no radius guides.
    for primitive in trajectory:
        if isinstance(primitive, (CurvilinearArcUnicycle, BackwardArc)):
            suffix = 'b' if isinstance(primitive, BackwardArc) else '0' if primitive is forward[0] else '1'
            arc_sector(ax, primitive, rf'$\mathbf o_{suffix}$', rf'$\iota_{suffix}$',
                       square=(family == 'CbCSC' and primitive is forward[0]))
            center = np.array([primitive.xc, primitive.yc])
            for point in ([primitive.x0, primitive.y0], [primitive.xf, primitive.yf]):
                line(ax, center, point, color=GREY, style=':', width=0.8)
        path(ax, primitive, PATH_COLOR, linewidth=2.5)
    heading(ax, START_POSITION, np.radians(initial_heading), START_POSE_COLOR)
    heading(ax, END_POSITION, np.radians(END_HEADING_DEGREES), END_POSE_COLOR)
    # Velocity arrows follow the path; heading arrows at the cusp describe the vehicle.
    heading(ax, q0, segment.theta0)
    heading(ax, qt, segment.theta0)
    initial_center = np.array([trajectory[0].xc, trajectory[0].yc])
    label_outside_circle(ax, START_POSITION, initial_center, r'$\mathbf p_0$', START_POSE_COLOR)
    label_outside_circle(ax, END_POSITION, ot, r'$\mathbf p_1$', END_POSE_COLOR)

    if family == 'CbCSC':
        backward = trajectory[0]
        ob = np.array([backward.xc, backward.yc])
        cusp = np.array([backward.xf, backward.yf])
        supporting_circle(ax, ob, r'$\mathcal O_b$', r'$\mathbf o_b$',
                          (-0.30, -0.60), color=GREY)
        heading(ax, cusp, backward.thetaf)
        label(ax, cusp, r'$\mathbf p_c$', (0, -5) if letter == 'd' else (0, 8),
              ha='center', va='top' if letter == 'd' else 'bottom')
        if tau0 == taut:
            line(ax, ob, ot, color=GREY, style='--', width=0.9)
        else:
            reflected = draw_reflection(ax, o0, ot, qt, direction, auxiliary=True)
            line(ax, ob, reflected, color=GREY, style='--', width=0.9)

    annotate_segment_length(ax, q0, qt, offset_ratio=0.45 if tau0 == taut else -0.45)

    initial_sign = '+' if tau0 > 0 else '-'
    target_sign = '+' if taut > 0 else '-'
    notation = rf'C^{{{initial_sign}}}SC^{{{target_sign}}}'
    if family == 'CbCSC':
        notation = rf'C^{{b{initial_sign}}}'+notation
    ax.text(0.02, 0.98, rf'{letter}) ${notation}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=20, zorder=25)
    ax.set_aspect('equal')
    ax.axis('off')
    print(f'{letter}) {family}: T={duration:.6f} s')
    return trajectory


def draw_reflection(ax, o0, ot, qt, direction, auxiliary):
    reflection = 2*np.outer(direction, direction)-np.eye(2)
    reflected = qt+reflection@(ot-qt)
    ax.add_patch(Circle(reflected, RADIUS, fill=False, color=AUXILIARY_COLOR,
                        linestyle='-.', lw=0.9, zorder=1))
    ax.plot(*reflected, 'o', ms=3, color=GREY)
    label(ax, reflected, r'$\widetilde{\mathbf o}_1$', (6, 4), color=GREY)
    if auxiliary:
        ax.add_patch(Circle(ot, 2*RADIUS, fill=False, color=AUXILIARY_COLOR,
                            linestyle='--', lw=0.9, zorder=0))
        label(ax, ot+np.array([1.0, 1.4]), r"$\mathcal{O}_1'$",
              (0, 0), color=GREY, ha='center', va='bottom')
    return reflected


def create_figure():
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7))
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.055,
                               top=0.88, wspace=0.04, hspace=0.14)
        configs = PANEL_CONFIGURATIONS
        for ax, config, letter in zip(axes.flat, configs, 'abcd'):
            draw_panel(ax, config, letter)
        # Identical scales, with each construction centered in its own panel.
        width = max(ax.dataLim.width for ax in axes.flat)*1.07
        geometry_height = max(ax.dataLim.height for ax in axes.flat)*1.07
        title_space = 0.55
        height = geometry_height+title_space
        for ax in axes.flat:
            cx = (ax.dataLim.x0+ax.dataLim.x1)/2
            cy = (ax.dataLim.y0+ax.dataLim.y1)/2+title_space/2
            ax.set_xlim(cx-width/2, cx+width/2)
            ax.set_ylim(cy-height/2, cy+height/2)
        # Fit the equal-aspect panels to the available grid cells, as in
        # the eight-panel construction figure.
        panel = axes[0, 0].get_position(original=True)
        figure.set_size_inches(11, 11*panel.width/panel.height*height/width)
        figure.canvas.draw()
        positions = [ax.get_position() for ax in axes.flat]
        divider_x = (positions[0].x1+positions[1].x0)/2
        divider_y = (positions[0].y0+positions[2].y1)/2-0.006
        figure.add_artist(Line2D(
            [divider_x, divider_x], [positions[-1].y0, positions[0].y1],
            transform=figure.transFigure, color='0.80', linewidth=0.4))
        figure.add_artist(Line2D(
            [0.065, 0.98], [divider_y, divider_y],
            transform=figure.transFigure, color='0.80', linewidth=0.4))
        return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            output = OUTPUT_DIRECTORY / f'candidate_family_constructions.{extension}'
            figure.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
