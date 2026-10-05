"""Explain exact curved waypoint-set propagation on five real corridors.

Adapted (with simplified coordinates) from the five-corridor random stress
sample, seed 37, accepted case 7. Run this file or use --no-show.
R_j here is named F_j in the implementation. Sets are computed analytically;
only their displayed outlines are sampled. Rightward entry and exit directions
include the first and last fillet constraints in A_1 and A_4, respectively.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from kappa_planner.baseline_construction import compute_filleted_baseline_exact
from kappa_planner.helpers.fillet_reachability import RoundedCornerConstraint, _make_set

OUTPUT = Path(__file__).parent / 'figures/planner_thesis'
R, r = 2., .5
CORRIDOR_BOUNDS = [
    (-7., -2.9, -1.3, .2),
    (-4.4, -2.9, -1.3, 4.5),
    (-4.2, 2.4, 2., 4.5),
    (-.1, 3., -.9, 4.2),
    (.2, 4.6, -2.3, 1.),
]
LOCAL, REACHABLE, EXCLUDED = '#CBE3D5', '#167DA7', '#A64A43'
INCOMING = '#B67D1C'
STYLE = {'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 17,
         'axes.titlesize': 19, 'axes.labelsize': 16, 'xtick.labelsize': 13,
         'ytick.labelsize': 13, 'legend.fontsize': 15}


def local_set(region, door):
    """A_j before any reachability clipping, rather than the rectangular F_j."""
    if region is None:
        return _make_set(door, (), 1e-9)
    bounds = (region['low'][0], region['high'][0],
              region['low'][1], region['high'][1])
    constraint = RoundedCornerConstraint(tuple(-region['offset']),
        tuple(map(int, region['frame'].sum(axis=1))), float(region['radius']))
    return _make_set(bounds, (constraint,), 1e-9)


def build_example():
    corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]]))
                 for a,b,c,d in CORRIDOR_BOUNDS]
    result = compute_filleted_baseline_exact(corridors, SimpleNamespace(r=r, R=R),
                                              initial_direction='right', final_direction='right')
    if not result.feasible or result.fillet_reachability is None:
        raise RuntimeError(f'Example failed: {result.status}')
    local = tuple(local_set(g, d) for g,d in
                  zip(result.fillet_regions, result.feasibility.safe_overlaps))
    reachable = result.fillet_reachability.reachable_sets
    # This cutoff is derived from the circle, not from a sampled outline.
    x_floor = -2.4-np.sqrt(1.5**2-1.**2)
    np.testing.assert_allclose(reachable[0].bounds[0], x_floor)
    floor = 4.-np.sqrt(1.5**2-(x_floor+4.9)**2)
    np.testing.assert_allclose(reachable[1].bounds[2], floor)
    np.testing.assert_allclose(reachable[2].bounds[2], floor)
    assert local[2].contains((1.6, 3.3)) and not reachable[2].contains((1.6, 3.3))
    assert not local[1].contains((-3.5, 3.3))
    assert all(s.contains(p) for s,p in zip(reachable, result.fillet_reachability.polyline))
    return result, local, reachable


def outline(region, count=601):
    """Vertical analytic slices produce a display polygon, never a solver input."""
    xs = np.linspace(region.bounds[0], region.bounds[1], count)
    slices = [region.slice_interval(0, x) for x in xs]
    valid = [(x, s) for x,s in zip(xs, slices) if s is not None]
    lower = [(x,s[0]) for x,s in valid]
    upper = [(x,s[1]) for x,s in reversed(valid)]
    return np.array(lower+upper)


def fill_set(ax, region, color, alpha=1., **kwargs):
    p = outline(region)
    return ax.fill(p[:,0], p[:,1], facecolor=color, alpha=alpha, **kwargs)[0]


def draw_sets(ax, a, reached, point=None):
    fill_set(ax, a, LOCAL, edgecolor='#57806B', linewidth=1.5, zorder=2)
    # Hatch exactly the difference of interval slices A_j \ R_j.
    xs = np.linspace(a.bounds[0], a.bounds[1], 901)
    lo, hi, rlo, rhi = [], [], [], []
    for x in xs:
        sa = a.slice_interval(0,x)
        if sa is None:
            lo.append(np.nan);hi.append(np.nan);rlo.append(np.nan);rhi.append(np.nan);continue
        sr = reached.slice_interval(0,x)
        lo.append(sa[0]);hi.append(sa[1]);rlo.append(sa[1] if sr is None else sr[0]);rhi.append(sa[1] if sr is None else sr[1])
    for bottom, top in ((lo,rlo),(rhi,hi)):
        ax.fill_between(xs,bottom,top,facecolor='none',edgecolor=EXCLUDED,
                        hatch='///',linewidth=0.,zorder=3)
    fill_set(ax, reached, REACHABLE, alpha=.7, edgecolor='#085577', linewidth=1.8,zorder=4)
    if point is not None:
        ax.plot(*point, 'o', color='#202020', ms=6,zorder=6)
    ax.grid(color='.88',linewidth=.6,zorder=0)
    ax.set_xlabel('$x$');ax.set_ylabel('$y$')


def draw_path(ax, result):
    points = result.fillet_reachability.polyline
    entry = np.array([CORRIDOR_BOUNDS[0][0]+r, points[0,1]])
    exit_point = np.array([CORRIDOR_BOUNDS[-1][1]-r, points[-1,1]])
    nominal = np.vstack((entry,points,exit_point))
    ax.plot(nominal[:,0],nominal[:,1],ls='--',color='.5',lw=1.1,zorder=4)
    first_tangent = result.fillets[0].incoming_tangent
    ax.plot([entry[0],first_tangent[0]], [entry[1],first_tangent[1]],
            color='#202020',lw=2.2,zorder=5)
    for j in range(len(points)-1):
        start = result.fillets[j].outgoing_tangent if result.fillets[j] else points[j]
        end = result.fillets[j+1].incoming_tangent if result.fillets[j+1] else points[j+1]
        ax.plot([start[0],end[0]],[start[1],end[1]],color='#202020',lw=2.2,zorder=5)
    last_tangent = result.fillets[-1].outgoing_tangent
    ax.plot([last_tangent[0],exit_point[0]], [last_tangent[1],exit_point[1]],
            color='#202020',lw=2.2,zorder=5)
    for f in result.fillets:
        if f is None:continue
        radial=f.incoming_tangent-f.center
        angles=np.arctan2(radial[1],radial[0])+np.linspace(0,f.signed_angle,150)
        arc=f.center+f.radius*np.column_stack((np.cos(angles),np.sin(angles)))
        ax.plot(arc[:,0],arc[:,1],color='#202020',lw=2.2,zorder=5)
    for j,p in enumerate(points):
        ax.plot(*p,'o',color='#202020',ms=5,zorder=6)
        ax.annotate('$\\mathbf{p}_{%d}$'%(j+1),p,xytext=(5,-15) if j==len(points)-1 else (5,7),
                    textcoords='offset points',fontsize=16,zorder=7)


def sequence_figure(result, local, reachable):
    fig=plt.figure(figsize=(13,12.3))
    gs=fig.add_gridspec(3,2,height_ratios=(1.05,1,1),hspace=.63,wspace=.3)
    overview=fig.add_subplot(gs[0,:])
    for j,(a,b,c,d) in enumerate(CORRIDOR_BOUNDS):
        overview.add_patch(Rectangle((a,c),b-a,d-c,facecolor='#E6EBEF',edgecolor='.5',alpha=.6,lw=1.1))
        labelpoints=[(-6.25,-.35),(-4.05,2.4),(-1.3,4.15),(2.35,2.),(3.6,-1.5)]
        overview.text(*labelpoints[j],'$\\mathcal{C}_{%d}$'%(j+1),fontsize=18)
    for a,s in zip(local,reachable):
        fill_set(overview,a,LOCAL,edgecolor='#57806B',lw=1.)
        fill_set(overview,s,REACHABLE,alpha=.8,edgecolor='#085577',lw=1.)
    draw_path(overview,result)
    overview.set_aspect('equal');overview.set_xlim(-7.6,5.1);overview.set_ylim(-2.8,5.1)
    overview.set_title('(a) Five corridors, four overlap sets; $R=2$, $r=0.5$',loc='left')
    overview.set_xlabel('$x$');overview.set_ylabel('$y$')
    notes=[
        '$\\mathcal{R}_1=\\mathcal{A}_1$\nRightward entry fillet included in $\\mathcal{A}_1$',
        '$\\mathcal{R}_2=\\mathcal{A}_2\\cap\\{x\\geq x_*,\\ y\\geq3.2\\}$\n$x_*=-3.51803\\ldots$; curved cutoff $y_*=3.41676\\ldots$',
        '$\\mathcal{R}_3=\\mathcal{A}_3\\cap\\{y_*\\leq y\\leq4,\\ x\\geq x_*+4\\}$\nThe round-corner cutoff $y_*=3.41676\\ldots$ is inherited',
        '$\\mathcal{R}_4=\\mathcal{A}_4\\cap\\{0.7\\leq x\\leq1.9,\\ y\\leq-0.3\\}$\nExit fillet included in $\\mathcal{A}_4$; both coordinates restricted',
    ]
    for j,(a,s) in enumerate(zip(local,reachable)):
        ax=fig.add_subplot(gs[1+j//2,j%2]);draw_sets(ax,a,s,result.fillet_reachability.polyline[j])
        dx=a.bounds[1]-a.bounds[0];dy=a.bounds[3]-a.bounds[2]
        ax.set_xlim(a.bounds[0]-.1*dx,a.bounds[1]+.1*dx)
        ax.set_ylim(a.bounds[2]-.1*dy,a.bounds[3]+.1*dy)
        ax.set_title('(%s) Overlap %d'%('bcde'[j],j+1),loc='left',y=1.30)
        ax.text(0,1.04,notes[j],transform=ax.transAxes,fontsize=13,va='bottom')
        if j==2:
            ax.plot(1.6,3.3,'x',color=EXCLUDED,ms=10,mew=2,zorder=6)
            ax.annotate('Locally admissible,\nunreachable',(1.6,3.3),xytext=(.65,2.7),fontsize=13,color=EXCLUDED,
                        arrowprops=dict(arrowstyle='->',color=EXCLUDED))
    fig.legend(handles=[Patch(facecolor=LOCAL,edgecolor='#57806B',label='$\\mathcal{A}_j$: locally admissible'),
                        Patch(facecolor=REACHABLE,alpha=.7,label='$\\mathcal{R}_j$: reachable from the first overlap'),
                        Patch(facecolor='none',edgecolor=EXCLUDED,hatch='///',label='$\\mathcal{A}_j\\setminus\\mathcal{R}_j$: excluded')],
               loc='lower center',bbox_to_anchor=(.5,.012),ncol=3,frameon=False,fontsize=13)
    fig.subplots_adjust(top=.90,bottom=.1,left=.08,right=.97)
    fig.suptitle('Exact propagation of rounded waypoint sets',y=.98,fontsize=22)
    return fig


def detail_figure(result, local, reachable):
    fig,axes=plt.subplots(1,2,figsize=(12.5,8.4))
    floor=reachable[1].bounds[2]
    x_floor=reachable[0].bounds[0]
    for j,ax in zip((1,2),axes):
        draw_sets(ax,local[j],reachable[j])
        ax.set_ylim(2.95,4.12)
        ax.axhline(floor,color=INCOMING,lw=1.7,ls='--',zorder=5)
        ax.axhline(3.2,color='.5',lw=1.3,ls=':',zorder=5)
        ax.axhline(3.3,color=EXCLUDED,lw=1.,ls=':',zorder=5)
    axes[0].set_xlim(-3.76,-3.34);axes[1].set_xlim(.3,2.03)
    axes[0].set_title('(a) Clip $\\mathcal{A}_2$ using $\\mathcal{R}_1$',loc='left')
    axes[1].set_title('(b) Propagate the exact $y$ range into $\\mathcal{A}_3$',loc='left')
    axes[0].axvline(x_floor,color=INCOMING,lw=1.5,ls='--',zorder=5)
    axes[0].plot(x_floor,floor,'o',color=INCOMING,ms=7,zorder=7)
    axes[0].plot(x_floor,3.3,'x',color=EXCLUDED,ms=11,mew=2,zorder=7)
    axes[1].plot(1.6,3.3,'x',color=EXCLUDED,ms=11,mew=2,zorder=7)
    axes[0].annotate('First attainable height',(x_floor,floor),xytext=(-3.73,3.83),fontsize=13,color=INCOMING,
                     arrowprops=dict(arrowstyle='->',color=INCOMING))
    axes[1].annotate('No predecessor at this height',(1.6,3.3),xytext=(.42,3.08),fontsize=13,color=EXCLUDED,
                     arrowprops=dict(arrowstyle='->',color=EXCLUDED))
    for ax,x in zip(axes,(-3.47,.75)):
        ax.plot(x,3.6,'o',color='#202020',ms=6,zorder=7)
    # Arrows connect panels at exactly the same world y coordinate.
    for y,color in ((3.6,'#202020'),(3.3,EXCLUDED)):
        axes[1].annotate('',xy=(.75 if y==3.6 else 1.6,y),xycoords=axes[1].transData,
                         xytext=(-3.47 if y==3.6 else x_floor,y),textcoords=axes[0].transData,
                         arrowprops=dict(arrowstyle='->',color=color,lw=1.3,ls='-' if y==3.6 else '--'),zorder=8)
    fig.text(.08,.235,'$\\mathcal{A}_2:\\ (x+4.9)^2+(4-y)^2\\leq1.5^2$\n'
             '$x_*=-2.4-\\sqrt{1.5^2-1^2}=-3.51803\\ldots$\n'
             '$y_* =4-\\sqrt{1.5^2-(x_*+4.9)^2}=3.41676\\ldots$',fontsize=17)
    fig.text(.08,.115,'The same $y$ is required along the rightward passage; $x_3-x_2\\geq2R=4$.\n'
             'The red point belongs to $\\mathcal{A}_3$, but every possible predecessor lies outside $\\mathcal{R}_2$.',fontsize=15)
    fig.legend(handles=[Line2D([],[],color=INCOMING,ls='--',label='Exact cutoff from the round corner'),
                        Line2D([],[],color='.5',ls=':',label='Rectangular-only cutoff: 3.2')],
               loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,frameon=False,fontsize=14)
    fig.subplots_adjust(top=.88,bottom=.48,left=.08,right=.97,wspace=.36)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    plt.rcParams.update(STYLE)
    result,local,reachable=build_example()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,fig in [('exact_set_propagation',sequence_figure(result,local,reachable)),
                     ('exact_set_propagation_detail',detail_figure(result,local,reachable))]:
        fig.savefig(OUTPUT/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(OUTPUT/f'{name}.png',dpi=180,bbox_inches='tight')
        print(OUTPUT/f'{name}.pdf')
    records=[]
    for j,(a,s) in enumerate(zip(local,reachable)):
        def describe(region):
            return dict(bounds=list(map(float,region.bounds)),constraints=[dict(
                center=list(map(float,g.center)),signs=g.signs,radius=g.radius) for g in region.constraints])
        records.append(dict(j=j+1,local=describe(a),reachable=describe(s)))
    data=dict(source='Simplified seed-37 five-corridor random stress case 7',R=R,r=r,
              corridors=CORRIDOR_BOUNDS,initial_direction='right',final_direction='right',
              directions=result.feasibility.passage_directions,
              sets=records,witness_polyline=result.fillet_reachability.polyline.tolist(),
              excluded_locally_admissible_point=[1.6,3.3])
    (OUTPUT/'exact_set_propagation.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Exact inherited y cutoff:',reachable[2].bounds[2])
    if not args.no_show:plt.show()


if __name__=='__main__':main()
