"""Two-panel corridor-sequence figure showing propagation and a filleted chain.

Run this file or use --no-show. Wider overlaps make the three extreme faces
visible at corridor scale. Disk erosion of the corridor union, local A_j, reachable
R_j and the validated filleted path are shown. Segment propagation computes
the sets; the generic solver independently checks the same reachable sets.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
import numpy as np

from example_exact_set_propagation import local_set, fill_set
from kappa_planner.baseline_construction import compute_filleted_baseline_exact, compute_filleted_baseline_segment
from kappa_planner.helpers.fillet_segment_reachability import extreme_segment_certificate, propagate_reachable_set
from kappa_planner.helpers.corridor_union import CorridorUnion
from kappa_planner.baseline_construction import _validate_and_build_fillets, _baseline_validation_key

OUTPUT=Path(__file__).parent/'figures/planner_thesis'
R,r=2.,.5
CORRIDOR_BOUNDS=[
    (-7.5,-2.4,-1.5,.7),
    (-4.8,-2.4,-1.5,4.5),
    (-4.8,2.2,2.,4.5),
    (-.5,2.2,-1.4,4.5),
    (-.5,6.,-1.4,1.3),
]
EROSION='#EDF5FC'
EROSION_EDGE='#C5DCEE'
POLYLINE='#A85521'
LOCAL,REACHABLE='#FBE1C3','#D5EDCF'
LOCAL_EDGE,REACHABLE_EDGE='#BA8047','#3C794A'
PROPAGATION='#B83131'


def draw_eroded_union(ax):
    """Plot U minus a radius-r disk using distance to U's exposed boundary.

    Exact finite-edge distances include circular offsets at reentrant corners.
    Only the display contour is sampled; no individually eroded rectangles are
    used, and these plotting samples do not affect reachable-set propagation.
    """
    union=CorridorUnion(CORRIDOR_BOUNDS)
    bounds=union.bounds
    x=np.linspace(bounds[:,0].min()-.1,bounds[:,1].max()+.1,1600)
    y=np.linspace(bounds[:,2].min()-.1,bounds[:,3].max()+.1,720)
    xx,yy=np.meshgrid(x,y)
    inside=np.zeros(xx.shape,dtype=bool)
    for a,b,c,d in bounds:
        inside|=(xx>=a)&(xx<=b)&(yy>=c)&(yy<=d)
    distance_squared=np.full(xx.shape,np.inf)
    for a,b in union.boundary:
        edge=b-a
        dx,dy=xx-a[0],yy-a[1]
        t=np.clip((dx*edge[0]+dy*edge[1])/(edge@edge),0,1)
        np.minimum(distance_squared,(dx-t*edge[0])**2+(dy-t*edge[1])**2,
                   out=distance_squared)
    clearance=np.where(inside,np.sqrt(distance_squared),-np.sqrt(distance_squared))
    ax.contourf(x,y,clearance,levels=[r,float(clearance.max())+1],
                colors=[EROSION],antialiased=True,zorder=0)
    ax.contour(x,y,clearance,levels=[r],colors=[EROSION_EDGE],linewidths=.7,zorder=1)
    ax.add_collection(LineCollection(union.boundary,colors='black',linewidths=.8,zorder=4))


def build_example():
    corridors=[SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]]))
               for a,b,c,d in CORRIDOR_BOUNDS]
    kwargs=dict(initial_direction='right',final_direction='right')
    robot=SimpleNamespace(r=r,R=R)
    result=compute_filleted_baseline_segment(corridors,robot,**kwargs)
    reference=compute_filleted_baseline_exact(corridors,robot,**kwargs)
    if not result.feasible or not reference.feasible:
        raise RuntimeError(f'Example failed: {result.status}, {reference.status}')
    reachable=result.fillet_reachability.reachable_sets
    local=tuple(local_set(g,d)for g,d in zip(result.fillet_regions,result.feasibility.safe_overlaps))
    steps=[]
    vectors={'right':np.array([1.,0.]),'left':np.array([-1.,0.]),
             'up':np.array([0.,1.]),'down':np.array([0.,-1.])}
    for j,(current,expected) in enumerate(zip(reachable,reference.fillet_reachability.reachable_sets)):
        np.testing.assert_allclose(current.bounds,expected.bounds,atol=1e-9,rtol=0)
        assert current.constraints==expected.constraints
        if j==len(reachable)-1:continue
        direction=result.feasibility.passage_directions[j]
        certificate=extreme_segment_certificate(current,direction)
        assert certificate.valid
        face=np.array(certificate.endpoints)
        shifted=face+2*R*vectors[direction]
        propagated=propagate_reachable_set(local[j+1],current,direction,R)
        np.testing.assert_allclose(propagated.bounds,reachable[j+1].bounds,atol=1e-9,rtol=0)
        assert propagated.constraints==reachable[j+1].constraints
        steps.append(dict(j=j+1,direction=direction,face=face,shifted=shifted))
    # Representative points are locally safe but lack a prefix.
    excluded=((1,(-3.5,2.9)),(2,(1.5,2.7)),(3,(1.2,.3)))
    for j,p in excluded:assert local[j].contains(p)and not reachable[j].contains(p)
    # Choose an interior witness for this illustration; propagation stays exact.
    points=np.array([[-4.05,-.7],[-4.05,3.65],[1.15,3.65],[1.15,-.65]])
    assert all(s.contains(p)for s,p in zip(reachable,points))
    validated=_validate_and_build_fillets(points,result.feasibility,result.fillet_regions,r,R,1e-9)
    if validated is None:raise RuntimeError('Interior illustration chain failed geometric validation.')
    fillets,remaining,violation=validated
    result=replace(result,polyline=points,fillets=fillets,remaining_lengths=remaining,
                   max_violation=violation,
                   fillet_reachability=replace(result.fillet_reachability,polyline=points),
                   _validation_key=_baseline_validation_key(points,result.feasibility,result.fillet_regions,r,R,1e-9))
    return result,local,reachable,steps


def draw_path(ax,result):
    points=result.polyline
    entry=np.array([CORRIDOR_BOUNDS[0][0]+r,points[0,1]])
    exit_point=np.array([CORRIDOR_BOUNDS[-1][1]-r,points[-1,1]])
    baseline=np.vstack((entry,points,exit_point))
    ax.plot(*baseline.T,color=POLYLINE,lw=1.5,ls=(0,(5,3)),zorder=4.5)
    ax.plot(*points.T,linestyle='none',marker='o',markersize=6,
            markerfacecolor='white',markeredgecolor=POLYLINE,markeredgewidth=1.5,zorder=7)
    pieces=[(entry,result.fillets[0].incoming_tangent)]
    pieces.extend((a.outgoing_tangent,b.incoming_tangent)for a,b in zip(result.fillets[:-1],result.fillets[1:]))
    pieces.append((result.fillets[-1].outgoing_tangent,exit_point))
    for a,b in pieces:ax.plot([a[0],b[0]],[a[1],b[1]],color='#252525',lw=2.,zorder=5)
    for f in result.fillets:
        radial=f.incoming_tangent-f.center
        angles=np.arctan2(radial[1],radial[0])+np.linspace(0,f.signed_angle,150)
        arc=f.center+R*np.column_stack((np.cos(angles),np.sin(angles)))
        ax.plot(arc[:,0],arc[:,1],color='#252525',lw=2.,zorder=5)


def draw_boundary_poses(ax,result):
    poses=[(CORRIDOR_BOUNDS[0][0]+r,result.polyline[0,1],'#16803a'),
           (CORRIDOR_BOUNDS[-1][1]-r,result.polyline[-1,1],'#dc2626')]
    for x,y,color in poses:
        ax.add_patch(Circle((x,y),r,fill=False,edgecolor=color,lw=1.3,zorder=8))
        ax.plot(x,y,marker='o',markersize=5,color=color,zorder=9)
        ax.arrow(x,y,.75,0,head_width=.20,head_length=.22,
                 length_includes_head=True,color=color,lw=1.3,zorder=9)


def leader(ax,text,anchor,label,color):
    ax.annotate(text,anchor,xytext=label,color=color,fontsize=34,
                arrowprops=dict(arrowstyle='-',color=color,lw=.9,shrinkA=3,shrinkB=2),zorder=9)


def draw_propagating_segments(ax,steps,local):
    for step in steps:
        face,shifted=step['face'],step['shifted']
        ax.plot(*face.T,color=PROPAGATION,lw=4.,zorder=6)
        ax.plot(*shifted.T,color=PROPAGATION,lw=1.8,ls=(0,(4,2)),zorder=6)
        # Beyond the translated segment, its endpoint rays bound the half-strip
        # intersected with the next local set. Show these through that set only.
        direction=step['direction']
        axis=0 if direction in ('right','left')else 1
        positive=direction in ('right','up')
        downstream=local[step['j']].bounds[2*axis+int(positive)]
        for start in shifted:
            end=start.copy()
            end[axis]=downstream
            if (end[axis]-start[axis])*(1 if positive else -1)>0:
                ax.plot([start[0],end[0]],[start[1],end[1]],color=PROPAGATION,
                        lw=.7,alpha=.8,zorder=4)
        midpoint=face.mean(axis=0)
        destination=shifted.mean(axis=0)
        ax.annotate('',destination,xytext=midpoint,
                    arrowprops=dict(arrowstyle='->',color=PROPAGATION,lw=.7),zorder=4)


def create_figure(result,local,reachable,steps):
    fig,axes=plt.subplots(1,2,figsize=(23.5,9.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.045,right=.99,top=.91,bottom=.34,wspace=.09)
    for panel,ax in enumerate(axes):
        draw_eroded_union(ax)
        draw_boundary_poses(ax,result)
        if panel==0:
            for a in local:
                fill_set(ax,a,LOCAL,edgecolor=LOCAL_EDGE,linewidth=1.3,zorder=2)
            leader(ax,r'$\mathcal{A}_2$',(-4.05,2.75),(-5.65,2.55),LOCAL_EDGE)
            leader(ax,r'$\mathcal{A}_3$',(1.45,2.65),(2.45,2.55),LOCAL_EDGE)
        else:
            for s in reachable:
                fill_set(ax,s,REACHABLE,edgecolor=REACHABLE_EDGE,linewidth=1.3,zorder=2)
        ax.set_xlim(-8.05,6.7);ax.set_ylim(-2.45,5.0);ax.set_aspect('equal')
        ax.set_xlabel('$x$');ax.grid(False)
    ax=axes[0]
    for s in reachable:
        fill_set(ax,s,REACHABLE,edgecolor=REACHABLE_EDGE,linewidth=1.3,zorder=3)
    draw_propagating_segments(ax,steps,local)
    leader(ax,r'$\mathcal{A}_1=\mathcal{R}_1$',(-3.65,-.65),(-5.9,-2.),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{R}_2$',(-3.5,3.8),(-5.65,3.65),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{R}_3$',(1.55,3.35),(2.5,3.85),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{A}_4$',(1.25,.2),(2.25,.65),LOCAL_EDGE)
    leader(ax,r'$\mathcal{R}_4$',(1.55,-.65),(2.45,-1.95),REACHABLE_EDGE)
    ax.set_ylabel('$y$')
    ax.set_title('a) Reachable-set propagation',fontsize=32,pad=20)
    ax=axes[1]
    draw_path(ax,result)
    leader(ax,r'$\mathcal{R}_1$',(-3.65,-.65),(-5.9,-2.),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{R}_2$',(-3.5,3.8),(-5.65,3.65),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{R}_3$',(1.55,3.35),(2.5,3.85),REACHABLE_EDGE)
    leader(ax,r'$\mathcal{R}_4$',(1.55,-.65),(2.45,-1.95),REACHABLE_EDGE)
    ax.set_title('b) Chosen polyline and filleted path',fontsize=32,pad=20)
    handles=[Patch(facecolor=LOCAL,edgecolor=LOCAL_EDGE,label=r'Locally admissible $\mathcal{A}_j$'),
             Patch(facecolor=REACHABLE,edgecolor=REACHABLE_EDGE,label=r'Reachable $\mathcal{R}_j$'),
             Line2D([],[],color='#252525',lw=2,label='Filleted path'),
             Line2D([],[],color=POLYLINE,lw=1.5,ls=(0,(5,3)),marker='o',
                    markerfacecolor='white',markeredgewidth=1.5,label='Chosen polyline and waypoints'),
             Line2D([],[],color=PROPAGATION,lw=4.,label='Origin segment'),
             Line2D([],[],color=PROPAGATION,lw=1.8,ls=(0,(4,2)),label=r'Segment shifted by $2R$')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,
               columnspacing=1.2,handlelength=1.7,handletextpad=.6,frameon=False,fontsize=30)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true');args=parser.parse_args()
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':30,
                         'xtick.labelsize':20,'ytick.labelsize':20})
    result,local,reachable,steps=build_example()
    fig=create_figure(result,local,reachable,steps)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png'):
        path=OUTPUT/f'segment_set_propagation.{extension}'
        fig.savefig(path,bbox_inches='tight',dpi=190);print(path)
    records=[dict(j=s['j'],direction=s['direction'],extreme_segment=s['face'].tolist(),
                  translated_segment=s['shifted'].tolist())for s in steps]
    data=dict(corridors=CORRIDOR_BOUNDS,R=R,r=r,algorithm=result.selection_method,
              generic_reference_agrees=True,steps=records,
              reachable_bounds=[list(map(float,s.bounds))for s in reachable],
              polyline=result.polyline.tolist())
    (OUTPUT/'segment_set_propagation.json').write_text(json.dumps(data,indent=2)+'\n')
    if not args.no_show:plt.show()


if __name__=='__main__':main()
