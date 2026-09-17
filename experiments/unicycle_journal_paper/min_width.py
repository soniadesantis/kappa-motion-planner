"""Minimum corridor widths, recreated from min_width_figure.py using kappa-planner."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc, Rectangle, ConnectionPatch
from kappa_planner.vehicle import Unicycle
from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector, get_corner_point
from tangent_convention_journal import STYLE
from example_one_corridor import arrow, text

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'


def build_geometry():
    vehicle = Unicycle(model='Rosbot circular')
    vehicle.width = 0.4
    vehicle.update(v_max=0.8, omega_max=1)
    corridors = [CorridorWorld(1.1, 3, [0, 0], np.pi/6)]
    for width, length, angle, extra in [(1.4, 2, np.pi/2, 2), (1.1, 3, -np.pi/6, 1.5)]:
        tail = np.array(corridors[-1].head)
        head = tail + length * np.array([np.cos(angle), np.sin(angle)])
        corridors.append(get_corridor_from_vector(tail, head, width, add_height=extra))
    planner = MotionPlanner(vehicle, corridors, assumptions='standing')
    minimum = [CorridorWorld(width, c.height, c.center, c.tilt)
               for width, c in zip(planner.min_corridor_widths, corridors)]
    fitted = MotionPlanner(vehicle, minimum, assumptions='standing')
    centers = [np.array([c.center.x, c.center.y]) for c in fitted.intermediate_circles]
    return vehicle, minimum, centers


def leader(ax, symbol, position, target, color='black', size=24, bend=0.2, label_gap=5):
    ax.annotate(symbol, xy=target, xytext=position, fontsize=size, color=color,
                ha='center', va='center', annotation_clip=False, zorder=12,
                arrowprops=dict(arrowstyle='<|-', color=color, lw=1.1, linestyle=':',
                                connectionstyle=f'arc3,rad={bend}', shrinkA=label_gap, shrinkB=2))


def draw_scene(ax, corridors, centers, R, r):
    for corridor in corridors:
        ax.add_patch(Polygon(corridor.corners, fill=False, edgecolor='black', lw=1.8))
    for i, center in enumerate(centers):
        ax.add_patch(Circle(center, R, fill=False, color='red', lw=1.5))
        ax.add_patch(Circle(center, R+r, fill=False, color='orange', lw=1.5))
        side = CorridorWorld.RGT if i == 0 else CorridorWorld.LFT
        normals = [c.outward_normals[side] for c in corridors[i:i+2]]
        angles = [np.degrees(np.arctan2(n[1], n[0])) for n in normals]
        # The active swept envelope joins the two outer boundary contacts.
        ax.add_patch(Arc(center, 2*(R+r), 2*(R+r), theta1=min(angles),
                         theta2=max(angles), color='#216C2D', lw=2.5, zorder=5))
        for normal in normals:
            point = center + (R+r) * normal
            ax.plot(*np.array([center, point]).T, color='black', lw=1.0)
            ax.plot(*point, 'o', color='#216C2D', ms=4, zorder=6)
        ax.plot(*center, 'ko', ms=5, zorder=9)


def dimension(ax, corridor, position, symbol, offset):
    normal = np.array(corridor.outward_normals[corridor.RGT])
    arrow(ax, position - corridor.width/2 * normal, position + corridor.width/2 * normal,
          style='<->', lw=1.2)
    text(ax, *(position + offset), symbol, size=23)


def create_figure():
    vehicle, corridors, centers = build_geometry()
    R, r = vehicle.max_radius, vehicle.width/2
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 8.6))
        ax = fig.add_axes((0.01, 0.04, 0.98, 0.95))
        draw_scene(ax, corridors, centers, R, r)
        for i, (center, offset) in enumerate(zip(centers, [(-0.30,0.56),(-0.53,-0.29)]), 1):
            text(ax, *(center + offset), rf'$\mathcal{{O}}_{i}$', size=25, color='red')
        leader(ax, r'$\mathbf{p}_1^o$', centers[0]+[0.32,0.40], centers[0])
        leader(ax, r'$\mathbf{p}_2^o$', centers[1]+[0.10,-0.44], centers[1])
        c = centers[0]
        ax.plot([c[0]-R-r,c[0]],[c[1],c[1]], color='orange', lw=1.1)
        text(ax, *(c+[-0.48,0.15]), r'$R+r$', color='orange', size=23)
        vec = np.array([np.cos(np.pi+0.2),np.sin(np.pi+0.2)])
        ax.plot(*np.array([c,c+R*vec]).T, color='red', lw=1.1)
        text(ax, *(c+[-0.64,-0.27]), r'$R$', color='red', size=23)
        for i, (corridor, fraction, offset) in enumerate(zip(corridors,[-0.7,-1.55,1.30],
                                                       [np.array([0.18,0.10]),np.array([0,-0.17]),np.array([0.20,0])]),1):
            pos = np.array(corridor.center) + fraction*np.array(corridor.unit_vector)
            dimension(ax,corridor,pos,rf'$\underline{{w}}_{i}$',offset)
        for pos, symbol in [((-1.05,-0.04),r'$\mathcal{C}_1$'),((1.91,0.05),r'$\mathcal{C}_2$'),
                             ((4.37,2.73),r'$\mathcal{C}_3$')]:
            text(ax,*pos,symbol,size=25)
        center = centers[1]
        ax.add_patch(Rectangle(center-[1.14,0.94],2.25,2.22,fill=False,edgecolor='0.65',lw=0.9))
        ax.set(xlim=(-1.65,5.2),ylim=(-1.15,4.60),aspect='equal')
        ax.axis('off')
        detail=fig.add_axes((0.57,0.035,0.415,0.43))
        draw_scene(detail,corridors,centers,R,r)
        detail.set(xlim=(center[0]-1.25,center[0]+1.13),
                   ylim=(center[1]-0.89,center[1]+1.22),aspect='equal')
        detail.set_xticks([])
        detail.set_yticks([])
        for spine in detail.spines.values():
            spine.set_color('0.65')
            spine.set_linewidth(0.9)
        corner=get_corner_point(corridors[1],corridors[2],-1)
        detail.plot(*corner,'o',color='red',ms=5,zorder=10)
        arrow(detail,center,corner,style='<->',color='#0072B2',lw=1.7,zorder=8)
        leader(detail,r'$R-r$',center+[0.02,0.66],corner+0.20*(center-corner),'#0072B2',bend=-0.2,label_gap=1)
        leader(detail,r'$\mathbf{p}_2^{\mathrm{corn}}$',corner+[-0.44,0.72],corner,'red')
        leader(detail,r'$\mathbf{p}_2^o$',center+[0.51,-0.29],center)
        text(detail,*(center+[-0.69,-0.16]),r'$\mathcal{O}_2$',size=24,color='red')
        corner_angle=np.degrees(np.arctan2(*(corner-center)[::-1]))
        for j, corridor in enumerate(corridors[1:]):
            normal=corridor.outward_normals[corridor.LFT]
            contact=center+(R+r)*normal
            projection=center+((corner-center)@normal)*normal
            arrow(detail,center,projection,style='<->',color='#00B44B',lw=1.5,zorder=9)
            arrow(detail,projection,contact,style='<->',lw=1.2,zorder=8)
            text(detail,*((center+projection)/2+([0,-0.12] if j==0 else [0.16,0])),r'$q_2$',color='#00B44B',size=23)
            text(detail,*((contact+projection)/2+([0,0.14] if j==0 else [0.17,0.04])),rf'$\underline{{w}}_{j+2}$',size=23)
            angle=np.degrees(np.arctan2(normal[1],normal[0]))
            detail.add_patch(Arc(center,0.38,0.38,theta1=min(angle,corner_angle),
                                 theta2=max(angle,corner_angle),color='black',lw=1))
            mid=np.radians((angle+corner_angle)/2)
            text(detail,*(center+0.30*np.array([np.cos(mid),np.sin(mid)])),r'$\beta_2$',size=21)
        fig.add_artist(ConnectionPatch(center+[1.11,-0.40],(0.31,1),
                       coordsA=ax.transData,coordsB=detail.transAxes,color='0.65',lw=0.9,zorder=0))
        return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    figure=create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for ext in ('pdf','png'):
            path=OUTPUT_DIRECTORY/f'min_width_recreated.{ext}'
            figure.savefig(path,dpi=300,bbox_inches='tight',pad_inches=0.04)
            print(f'Saved {path}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__=='__main__':
    main()
