"""Four vertical CSC/TCSC constructions for the journal paper.

Edit PANEL_CONFIGURATIONS to adjust boundary headings. Positions and radius
are shared, and all paths are computed by kappa-planner.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Wedge, FancyArrowPatch
from kappa_planner.geometry import Pose, Point
from kappa_planner.vehicle import Unicycle
from kappa_planner.trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle, TurnOnTheSpot
from kappa_planner.helpers.pose_to_pose_unicycle import compute_CSC_trajectory, compute_TCSC_trajectory
from tangent_convention_journal import STYLE
from main_example import path

OUTPUT_DIRECTORY=Path(__file__).resolve().parent/'figures'
RADIUS=0.85
START_POSITION=np.array([0.,0.])
END_POSITION=np.array([0.,4.5])
PANEL_CONFIGURATIONS=[('CSC',1,1,30),('TCSC',1,1,-60),
                      ('CSC',-1,1,150),('TCSC',-1,1,-145)]
BLUE='#189DE8'
GREY='0.55'


def label(ax,point,symbol,offset=(0,0),size=14,color='black',**kwargs):
    ax.text(*(np.asarray(point)+offset),symbol,fontsize=size,color=color,
            ha='center',va='center',zorder=12,**kwargs)


def heading(ax,point,angle,color='black',length=0.34):
    ax.plot(*point,'o',color=color,ms=3.5,zorder=10)
    ax.add_patch(FancyArrowPatch(point,np.asarray(point)+length*np.array([np.cos(angle),np.sin(angle)]),
                 arrowstyle='-|>',mutation_scale=11,color=color,lw=1.2,
                 shrinkA=0,shrinkB=0,zorder=11))


def angular_reference(ax,vertex,start,end,radius,symbol,color=GREY,shade=False,label_offset=(0,0)):
    low,high=sorted(np.degrees([start,end]))
    if shade:
        ax.add_patch(Wedge(vertex,radius,low,high,facecolor=color,alpha=0.15,edgecolor='none'))
    ax.add_patch(Arc(vertex,2*radius,2*radius,theta1=low,theta2=high,color=color,lw=0.9,zorder=5))
    mid=(start+end)/2
    label(ax,np.asarray(vertex)+(radius+0.14)*np.array([np.cos(mid),np.sin(mid)]),symbol,
          offset=label_offset,size=13,color=color)


def build_panel(config):
    family,tau0,tauf,start_heading=config
    vehicle=Unicycle(model='Rosbot circular')
    vehicle.update(v_max=RADIUS,omega_max=1)
    start=Pose(Point(*START_POSITION),np.radians(start_heading))
    end=Pose(Point(*END_POSITION),np.pi)
    compute=compute_CSC_trajectory if family=='CSC' else compute_TCSC_trajectory
    trajectory,time=compute(start,end,tau0,tauf,vehicle)
    return trajectory,time


def draw_panel(ax,config,letter):
    family,tau0,tauf,_=config
    trajectory,time=build_panel(config)
    arcs=[p for p in trajectory if isinstance(p,CurvilinearArcUnicycle)]
    segment=next(p for p in trajectory if isinstance(p,LinearSegmentUnicycle))
    centers=[np.array([arc.xc,arc.yc]) for arc in arcs]
    q0=np.array([segment.x0,segment.y0]); qf=np.array([segment.xf,segment.yf])
    direction=(qf-q0)/np.linalg.norm(qf-q0)
    tangent_angle=np.arctan2(direction[1],direction[0])
    signs=f'{"+" if tau0>0 else "-"},{"+" if tauf>0 else "-"}'
    tangent_end=qf+1.25*direction
    ax.plot(*np.array([q0-0.80*direction,tangent_end]).T,'--',color=GREY,lw=0.8)
    label(ax,tangent_end,rf'$t^{{{signs}}}$',offset=(0.26,0.02),size=14)
    for i,(center,arc) in enumerate(zip(centers,arcs)):
        ax.add_patch(Circle(center,RADIUS,fill=False,edgecolor=GREY,lw=0.8,linestyle='--'))
        heading_point=np.array([arc.x0,arc.y0])
        end_point=np.array([arc.xf,arc.yf])
        for point in (heading_point,end_point):
            ax.plot(*np.array([center,point]).T,':',color=GREY,lw=0.8)
        ax.plot(*center,'o',color='0.25',ms=3,zorder=8)
        center_label_offset = ((0.43,-0.43) if i==0 and tau0!=tauf
                               else (-0.38,-0.40 if i==0 else 0.25))
        ax.annotate(rf'$\mathbf{{p}}_{"0" if i==0 else "1"}^o$',
                    xy=center, xytext=center+center_label_offset,
                    fontsize=14, ha='center', va='center', zorder=12,
                    arrowprops=dict(arrowstyle='<|-', linestyle='--', color='black',
                                    lw=0.8, connectionstyle='arc3,rad=0.15',
                                    relpos=(0.65,0.55 if i==1 else 0.20),
                                    shrinkA=0, shrinkB=1))
        circle_label_offset = ((0.60,0.32) if tau0!=tauf else (-0.60,0.32)) if i==0 else (-0.56,-0.34)
        if letter in ('c', 'd') and i == 0:
            circle_label_offset = (0.68,0)
        label(ax,center,rf'$\mathcal{{O}}_{i}$',offset=circle_label_offset,size=16)
        radial0=np.arctan2(*(heading_point-center)[::-1])
        sweep=arc.turn_direction*((arc.turn_direction*(np.arctan2(*(end_point-center)[::-1])-radial0))%(2*np.pi))
        arc_index=trajectory.index(arc)+1
        if family == 'TCSC' and i == 0:
            first_radius = (heading_point-center)/RADIUS
            last_radius = (end_point-center)/RADIUS
            marker_size = 0.20
            square = np.array([center,
                               center+marker_size*first_radius,
                               center+marker_size*(first_radius+last_radius),
                               center+marker_size*last_radius])
            ax.fill(*square.T,color='#AA4499',alpha=0.15,edgecolor='none',zorder=4)
            ax.plot(*square[1:].T,color='#AA4499',lw=0.9,zorder=5)
            middle = radial0+sweep/2
            label(ax,center+0.39*np.array([np.cos(middle),np.sin(middle)]),
                  rf'$\iota_{arc_index}$',size=13,color='#AA4499')
        else:
            angular_reference(ax,center,radial0,radial0+sweep,0.25,rf'$\iota_{arc_index}$',
                              color='#AA4499',shade=True)
    alpha=np.arctan2(*(centers[1]-START_POSITION)[::-1])
    ax.plot(*np.array([START_POSITION,centers[1]]).T,'--',color=GREY,lw=0.8)
    vertex=START_POSITION+np.array([0,2.35])
    ax.plot(*np.array([vertex,vertex+[0.40,0]]).T,color=GREY,lw=0.8)
    angular_reference(ax,vertex,0,alpha,0.25,r'$\alpha_0$')
    if tau0!=tauf:
        # Reflect the final circle and arc across their supporting tangent.
        reflection=2*np.outer(direction,direction)-np.eye(2)
        reflected_center=qf+reflection@(centers[1]-qf)
        ax.plot(*reflected_center,'o',color='0.55',ms=3,zorder=8)
        ax.add_patch(Circle(reflected_center,RADIUS,fill=False,color='0.65',lw=0.8,linestyle='--'))
        last=arcs[-1]
        a=np.arctan2(last.y0-last.yc,last.x0-last.xc)
        b=np.arctan2(last.yf-last.yc,last.xf-last.xc)
        sweep=last.turn_direction*((last.turn_direction*(b-a))%(2*np.pi))
        values=np.linspace(a,a+sweep,100)
        samples=centers[1]+RADIUS*np.array([np.cos(values),np.sin(values)]).T
        reflected=qf+(samples-qf)@reflection.T
        ax.plot(*reflected.T,'--',color=BLUE,lw=2.0,zorder=4)
        final_direction=reflection@np.array([-1.,0.])
        heading(ax,reflected[-1],np.arctan2(final_direction[1],final_direction[0]),'0.65')
        label(ax,reflected[-1],r'$\mathbf{x}_t^{\mathrm{ref}}$',offset=(0.12,0.20),size=13,color='0.4')
        label(ax,reflected_center,r'$\mathcal{O}_1^{\mathrm{ref}}$',offset=(0.45,-0.40),size=15,color='0.4')
        ax.plot(*np.array([START_POSITION,reflected_center]).T,'--',color=GREY,lw=0.9,zorder=2)
        reflected_direction=np.arctan2(*(reflected_center-START_POSITION)[::-1])
        angular_reference(ax,START_POSITION,reflected_direction,alpha,2.05,r"$\beta'$",shade=False,
                          label_offset=(0.18,0))
    beta=np.arcsin(RADIUS/np.linalg.norm(centers[1]-START_POSITION))
    # Point-to-circle tangent on the side consistent with the final turn.
    beta_direction = alpha - tauf * beta
    tangent_unit = np.array([np.cos(beta_direction), np.sin(beta_direction)])
    tangent_length = np.sqrt(np.sum((centers[1]-START_POSITION)**2)-RADIUS**2)
    beta_contact = START_POSITION + tangent_length * tangent_unit
    ax.plot(*np.array([START_POSITION,beta_contact]).T,'--',color=GREY,lw=0.9,zorder=2)
    angular_reference(ax,START_POSITION,beta_direction,alpha,1.85,r'$\beta$')
    for primitive in trajectory:
        path(ax,primitive,BLUE,linewidth=2.6)
    # Geometric straight-segment length, placed along the circle-center reference.
    dim_start=centers[0]
    dim_end=centers[1] if tau0==tauf else reflected_center
    dimension_vector=dim_end-dim_start
    dimension_normal=np.array([-dimension_vector[1],dimension_vector[0]])
    dimension_normal/=np.linalg.norm(dimension_normal)
    dimension_offset=(0.18 if tau0==tauf else -0.25)*dimension_normal
    dim_start=dim_start+dimension_offset
    dim_end=dim_end+dimension_offset
    ax.add_patch(FancyArrowPatch(dim_start,dim_end,arrowstyle='<->',mutation_scale=9,
                                 color=GREY,lw=0.8,shrinkA=0,shrinkB=0,zorder=3))
    label(ax,(dim_start+dim_end)/2,rf'$d_{trajectory.index(segment)+1}$',
          offset=(-0.15,0) if tau0==tauf else -0.18*dimension_normal,size=14)
    for i,p in enumerate(trajectory):
        heading(ax,[p.x0,p.y0],p.theta0)
        if isinstance(p,TurnOnTheSpot):
            angular_reference(ax,START_POSITION,p.theta0,p.thetaf,0.32,r'$\phi_1$',color='#A16B00',shade=True)
    heading(ax,END_POSITION,np.pi)
    if letter in ('a', 'c'):
        label(ax,START_POSITION,r'$\mathbf{x}_1^s\equiv\mathbf{x}_0$',offset=(0.08,-0.19),size=13)
        label(ax,END_POSITION,r'$\mathbf{x}_3^e\equiv\mathbf{x}_t$',offset=(0,0.18),size=13)
        label(ax,q0,r'$\mathbf{x}_1^e\equiv\mathbf{x}_2^s$',
              offset=(-0.43,-0.04) if letter=='c' else (0.43,-0.04),size=13)
        label(ax,qf,r'$\mathbf{x}_2^e\equiv\mathbf{x}_3^s$',offset=(0.43,-0.05),size=13)
    elif letter in ('b', 'd'):
        label(ax,START_POSITION,r'$\mathbf{x}_1^s\equiv\mathbf{x}_0$',
              offset=(-0.12,-0.33) if letter=='d' else (-0.12,-0.43),size=13)
        label(ax,START_POSITION,r'$\mathbf{x}_1^e\equiv\mathbf{x}_2^s$',
              offset=(-0.43,0.26) if letter=='d' else (0.28,0.19),size=13)
        label(ax,q0,r'$\mathbf{x}_2^e\equiv\mathbf{x}_3^s$',
              offset=(-0.43,-0.04) if letter=='d' else (0.43,-0.04),size=13)
        label(ax,qf,r'$\mathbf{x}_3^e\equiv\mathbf{x}_4^s$',offset=(0.43,-0.05),size=13)
        label(ax,END_POSITION,r'$\mathbf{x}_4^e\equiv\mathbf{x}_t$',offset=(0,0.18),size=13)
    else:
        label(ax,START_POSITION,r'$\mathbf{x}_0$',offset=(-0.30,-0.24),size=15)
        label(ax,END_POSITION,r'$\mathbf{x}_t$',offset=(0,0.23),size=15)
        label(ax,q0,rf'$\mathbf{{x}}_{trajectory.index(segment)+1}^s$',offset=(0.28,-0.05),size=13)
        label(ax,qf,rf'$\mathbf{{x}}_{trajectory.index(segment)+1}^e$',offset=(0.28,-0.08),size=13)
    notation=[]
    for p in trajectory:
        if isinstance(p,LinearSegmentUnicycle): notation.append('S')
        else: notation.append(('T' if isinstance(p,TurnOnTheSpot) else 'C')+('^{+}' if p.turn_direction>0 else '^{-}'))
    for symbol in ax.texts:
        symbol.set_fontsize(symbol.get_fontsize() * 1.15)
    ax.set_title(f'{letter})  $'+''.join(notation)+'$',fontsize=18,pad=3)
    ax.set_aspect('equal'); ax.axis('off')
    return time


def create_figure():
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,4,figsize=(14.5,7.4))
        fig.subplots_adjust(left=0.015,right=0.995,bottom=0.06,top=0.88,wspace=-0.04)
        for ax,config,letter in zip(axes,PANEL_CONFIGURATIONS,'abcd'):
            time=draw_panel(ax,config,letter)
            ax.set_ylim(-0.68,4.99)
            ax.set_xlim((-2.30,1.90) if config[1]==config[2] else (-1.30,2.90))
            print(f'{letter}) {config[0]} ({config[1]:+d},{config[2]:+d}): T={time:.6f}')
        first_position = axes[0].get_position()
        axes[0].set_position([first_position.x0 + 0.025, first_position.y0,
                              first_position.width, first_position.height])
        return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    figure=create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for ext in ('pdf','png'):
            output=OUTPUT_DIRECTORY/f'trajectory_constructions.{ext}'
            figure.savefig(output,dpi=300,bbox_inches='tight',pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show: plt.close(figure)
    else: plt.show()


if __name__=='__main__':
    main()
