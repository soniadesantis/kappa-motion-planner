"""Maximum intermediate-circle shifts, migrated from max_moving_circle.py."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from tangent_convention_journal import STYLE
from example_one_corridor import text

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'


def build_problem():
    vehicle = Unicycle(model='Rosbot circular')
    vehicle.width = 0.4
    vehicle.update(v_max=0.8, omega_max=1)
    corridors = [CorridorWorld(1.1, 3, [0, 0], np.pi/6)]
    for width, length, angle, extra in [(1.3, 2, np.pi/2, 2), (1.5, 3, -np.pi/6, 1.5)]:
        tail = np.array(corridors[-1].head)
        head = tail + length*np.array([np.cos(angle), np.sin(angle)])
        corridors.append(get_corridor_from_vector(tail, head, width, add_height=extra))
    planner = MotionPlanner(vehicle, corridors, assumptions='standing')
    for circle in planner.intermediate_circles:
        circle.update_s(circle.s_max)
    return planner


def center_label(ax, index, center, position, shifted, color, bend):
    state = rf'$s_{index}=s_{{{index},\mathrm{{max}}}}$' if shifted else rf'$s_{index}=0$'
    label = rf'$\mathbf{{p}}_{index}^o$' + '\n' + state
    if shifted and index == 1:
        ax.text(*position, label, color=color, fontsize=24,
                ha='center', va='center', linespacing=1.25, zorder=12)
        ax.add_patch(FancyArrowPatch(center, np.asarray(position) + [0.18, 0.10],
                     arrowstyle='-|>', color=color, lw=1.3, mutation_scale=13,
                     linestyle=':', connectionstyle=f'arc3,rad={-bend}',
                     shrinkA=3, shrinkB=0, zorder=12))
        return
    ax.annotate(label, xy=center, xytext=position, color=color, fontsize=24,
                ha='center', va='center', linespacing=1.25, zorder=12,
                arrowprops=dict(arrowstyle='<|-', color=color, lw=1.3,
                                linestyle=':', connectionstyle=f'arc3,rad={bend}',
                                shrinkA=1 if shifted and index == 1 else 6, shrinkB=3))


def create_figure():
    planner = build_problem()
    R, r = planner.vehicle.max_radius, planner.vehicle.width/2
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 8.4))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
        for corridor in planner.corridor_list:
            ax.add_patch(Polygon(corridor.corners, fill=False, edgecolor='black', lw=2))
        for index, circle in enumerate(planner.intermediate_circles, 1):
            original = np.array([circle.canonical_center.x, circle.canonical_center.y])
            shifted = np.array([circle.center.x, circle.center.y])
            for radius in (R, R+r):
                ax.add_patch(Circle(original, radius, fill=False, color='0.55', lw=1.5, linestyle='--'))
            ax.add_patch(Circle(shifted, R, fill=False, color='red', lw=1.7))
            ax.add_patch(Circle(shifted, R+r, fill=False, color='orange', lw=1.7))
            ax.plot(*np.array([original,shifted]).T, color='black', lw=1.3)
            ax.plot(*original, 'o', color='0.55', ms=6, zorder=10)
            ax.plot(*shifted, 'ko', ms=6, zorder=10)
            if index == 1:
                center_label(ax,index,original,original+[-1.42,0.59],False,'0.5',-0.25)
                center_label(ax,index,shifted,shifted+[-1.14,-0.98],True,'black',0.30)
                text(ax,*(shifted+[0.35,-0.39]),r'$\mathcal{O}_1$',size=27,color='red')
            else:
                center_label(ax,index,original,original+[1.06,-0.93],False,'0.5',0.30)
                center_label(ax,index,shifted,shifted+[1.88,0.99],True,'black',-0.30)
                text(ax,*(shifted+[-0.34,0.48]),r'$\mathcal{O}_2$',size=27,color='red')
                ax.plot([shifted[0]-R-r,shifted[0]],[shifted[1],shifted[1]],color='orange',lw=1.1)
                text(ax,*(shifted+[-0.36,0.15]),r'$R+r$',color='orange',size=25)
                radial=R*np.array([np.cos(np.pi+0.19),np.sin(np.pi+0.19)])
                ax.plot(*np.array([shifted,shifted+radial]).T,color='red',lw=1.1)
                text(ax,*(shifted+[-0.51,-0.28]),r'$R$',color='red',size=25)
        ax.set(xlim=(-1.95,5.20),ylim=(-1.43,4.82),aspect='equal')
        ax.axis('off')
        return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    figure=create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf','png'):
            output=OUTPUT_DIRECTORY/f'max_moving_recreated.{extension}'
            figure.savefig(output,dpi=300,bbox_inches='tight',pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__=='__main__':
    main()
