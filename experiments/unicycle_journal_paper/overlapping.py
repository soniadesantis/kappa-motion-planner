"""Overlapping-circle example, migrated from overlapping_circles.py."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Unicycle
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle, TurnOnTheSpot
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.poses import pose_from_shrunken_corridor_relative_frame
from tangent_convention_journal import STYLE
from main_example import path, pose
from example_one_corridor import text

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'


def build_problem():
    vehicle=Unicycle(model='Rosbot circular')
    vehicle.update(v_max=0.8,omega_max=1)
    r=vehicle.width/2
    # The original script establishes poses in these reference corridors first.
    reference=[get_corridor_from_vector([0,0],[0,3.5],1.5,add_height=0.7),
               get_corridor_from_vector([0,3.5],[3.5,3.5],1.5,add_height=0.7)]
    start=pose_from_shrunken_corridor_relative_frame([0.7,-0.2,-np.pi/6],reference[0].shrink(r))
    end=pose_from_shrunken_corridor_relative_frame([0.45,0.3,5*np.pi/4],reference[1].shrink(r))
    corridors=[CorridorWorld(1.5,3.5,[0,2.5],np.pi/2),
               CorridorWorld(1.5,3.5,[1,3.5],0)]
    planner=MotionPlanner(vehicle,corridors,start_pose=start,end_pose=end,assumptions='standing')
    trajectory=planner.compute_trajectory_analytical()
    # Tangent circles meet without a finite straight segment between them.
    visible=[p for p in trajectory if not (isinstance(p,LinearSegmentUnicycle) and p.path_length<1e-9)]
    return planner,visible


def create_figure():
    planner,trajectory=build_problem()
    r=planner.vehicle.width/2
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(8.5,8.0))
        fig.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98)
        for corridor in planner.corridor_list:
            ax.add_patch(Polygon(corridor.corners,fill=False,edgecolor='black',lw=1.6))
            ax.add_patch(Polygon(corridor.shrink(r).corners,fill=False,
                                 edgecolor='black',lw=1.4,linestyle='--'))
        arcs=[p for p in trajectory if isinstance(p,CurvilinearArcUnicycle)]
        for arc,name,offset in zip(arcs,['0','1','f'],[(-0.10,-0.60),(0.48,-0.27),(-0.54,0.05)]):
            center=np.array([arc.xc,arc.yc])
            ax.add_patch(Circle(center,arc.radius,fill=False,
                                color='red' if name=='1' else 'black',lw=0.9,linestyle='--'))
            ax.plot(*center,'ko',ms=2.5)
            text(ax,*(center+offset),rf'$\mathcal{{O}}_{name}$',size=36)
        for primitive in trajectory:
            path(ax,primitive,'black',linewidth=4.2)
        poses=[planner.start_pose]+[[p.xf,p.yf,p.thetaf] for p in trajectory]
        seen=set()
        for x,y,heading in poses:
            key=tuple(np.round([x,y,heading],7))
            if key not in seen:
                pose(ax,(x,y),heading,r)
                seen.add(key)
        symbols=[]
        for i,primitive in enumerate(trajectory,1):
            name='T' if isinstance(primitive,TurnOnTheSpot) else 'C' if isinstance(primitive,CurvilinearArcUnicycle) else 'S'
            symbol=rf'{name}_{{{i}}}'
            if name!='S':
                sign='+' if primitive.turn_direction>0 else '-'
                symbol+=rf'^{{{sign}}}'
            symbols.append(symbol)
        text(ax,1.90,1.06,'$'+r'\,'.join(symbols)+'$',size=28)
        ax.set(xlim=(-1.18,3.08),ylim=(0.59,4.72),aspect='equal')
        ax.axis('off')
        return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    figure=create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for ext in ('pdf','png'):
            output=OUTPUT_DIRECTORY/f'overlapping_recreated.{ext}'
            figure.savefig(output,dpi=300,bbox_inches='tight',pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__=='__main__':
    main()
