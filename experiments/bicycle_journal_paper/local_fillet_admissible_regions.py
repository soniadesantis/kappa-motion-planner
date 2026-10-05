"""Local A_j waypoint regions in the canonical L-junction, for the thesis.

Run with VS Code's Run button or --no-show. Two panels show the region and
a feasible waypoint with its fillet. The single shaded A_j comprises the
local safety OR rule after tangent containment. These are WAYPOINT regions,
not center regions: for this downward-to-right turn o = p + (R,R).
The circular boundary is consequently centered at (-R,-R) in waypoint space,
whereas the orthonormal frame remains attached to the physical corner (0,0).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as PlotPath
import numpy as np

from corridor_dimensions_study import STYLE
from kappa_planner.helpers.fillet_safety import fillet_vertex_region, region_contains_point


OUTPUT_DIRECTORY = Path(__file__).parent / 'figures/planner_thesis'
R, r = 2., .4
WIDTH_X, WIDTH_Y = 4.5, 4.2
ADMISSIBLE_COLOR = '#B4DFCA'
THRESHOLD_COLOR = '#806493'
FIGURE_STYLE = {**STYLE, 'text.latex.preamble': r'\usepackage{amsmath}\usepackage{bm}'}


def local_region():
    """Use the planner's original local model, with full safe overlap bounds."""
    door = dict(x=(-WIDTH_X+r, -r), y=(-WIDTH_Y+r, -r))
    region = fillet_vertex_region(door, (), np.zeros(2), [0., -1.], [1., 0.], r, R,
                                 pair_bounds=((-WIDTH_X, 0., -WIDTH_Y, 6.),
                                              (-WIDTH_X, 6., -WIDTH_Y, 0.)))
    np.testing.assert_allclose(region['low'], [door['x'][0], door['y'][0]])
    np.testing.assert_allclose(region['high'], [-r, -r])
    # Check the curved boundary and representative points in each colored part.
    for angle in np.linspace(0, np.pi/2, 41):
        point = -R+(R-r)*np.array([np.cos(angle), np.sin(angle)])
        if not region_contains_point(point, region):
            raise RuntimeError('Displayed circular boundary differs from the planner A_j.')
    for point in ((-3., -1.), (-1., -3.), (-1.2, -1.2)):
        if not region_contains_point(point, region):
            raise RuntimeError('Displayed admissible point fails original local model.')
    if region_contains_point((-.6, -.6), region):
        raise RuntimeError('Expected excluded portion of D_j is admissible.')
    return door


def draw_region(ax, detailed=True):
    door = local_region()
    xmin, xmax = door['x']
    ymin, ymax = door['y']
    extent = 6.
    # Same white corridors and rounded light-blue erosion as local_l_junction.
    arc = PlotPath.arc(180, 270)
    vertices = [(xmin,ymin), (xmin,extent-r), (-r,extent-r), (-r,0)]
    codes = [PlotPath.MOVETO]+[PlotPath.LINETO]*3
    vertices.extend(r*arc.vertices[1:])
    codes.extend(arc.codes[1:])
    vertices.extend([(extent-r,-r), (extent-r,ymin), (xmin,ymin)])
    codes.extend([PlotPath.LINETO, PlotPath.LINETO, PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(PlotPath(vertices,codes),facecolor='#EDF5FC',
                           edgecolor='#486F91',linewidth=.7,zorder=1))
    for bounds in ((-WIDTH_X,0.,-WIDTH_Y,extent), (-WIDTH_X,extent,-WIDTH_Y,0.)):
        a,b,c,d = bounds
        ax.add_patch(Rectangle((a,c),b-a,d-c,fill=False,edgecolor='.25',linewidth=.5,zorder=2))
    # Disjoint partition: the first branch includes the shared nonpositive
    # quadrant; the second includes only o_x>0, preventing double coloring.
    ax.add_patch(Rectangle((xmin,ymin),-R-xmin,ymax-ymin,
                           facecolor=ADMISSIBLE_COLOR,edgecolor='none',zorder=3))
    ax.add_patch(Rectangle((-R,ymin),xmax+R,-R-ymin,
                           facecolor=ADMISSIBLE_COLOR,edgecolor='none',zorder=3))
    ax.add_patch(Wedge((-R,-R),R-r,0,90,facecolor=ADMISSIBLE_COLOR,edgecolor='none',zorder=3))
    # Complete A_j contour, with a translated quarter-circular boundary.
    arc = PlotPath.arc(0,90)
    vertices = [(xmin,ymin), (xmax,ymin), (xmax,-R)]
    codes = [PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO]
    vertices.extend((R-r)*arc.vertices[1:]+(-R,-R))
    codes.extend(arc.codes[1:])
    vertices.extend([(xmin,ymax),(xmin,ymin)])
    codes.extend([PlotPath.LINETO,PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(PlotPath(vertices,codes),fill=False,
                           edgecolor='#444444',linewidth=1.,zorder=4))
    ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,
                           edgecolor='#0072B2',linewidth=1.,linestyle='--',zorder=5))
    if detailed:
        ax.plot([-R,-R],[ymin,.65],color=THRESHOLD_COLOR,lw=1.3,ls=':',zorder=5)
        ax.plot([xmin,1.2],[-R,-R],color=THRESHOLD_COLOR,lw=1.3,ls=':',zorder=5)
        ax.text(-R-.13,.65,r'$\bar p_{x,j}=-R$',ha='right',va='bottom',fontsize=28,
                color=THRESHOLD_COLOR)
        ax.text(1.8,-R+.1,r'$\bar p_{y,j}=-R$',ha='right',va='bottom',fontsize=28,
                color=THRESHOLD_COLOR)
        ax.text(-3.1,-2.65,r'$\mathcal A_j$',ha='center',va='center',fontsize=36)
        ax.annotate(r'$\mathcal D_j$',xy=(xmax,-.65),xytext=(.7,-.9),
                    ha='left',va='center',fontsize=36,color='#0072B2',
                    arrowprops=dict(arrowstyle='->',lw=.8,color='#0072B2',shrinkB=0),zorder=7)
        # Radius refers to the waypoint-space circular boundary, not an arc path.
        radial = np.array([-R,-R])+(R-r)*np.ones(2)/np.sqrt(2)
        ax.add_patch(FancyArrowPatch((-R,-R),radial,arrowstyle='->',
                                    mutation_scale=17,lw=1.1,color='.3',shrinkA=0,shrinkB=0,zorder=6))
        ax.annotate(r'$R-r$',xy=(-R+(R-r)/np.sqrt(2)/2,)*2,
                    xytext=(24,0),textcoords='offset points',rotation=45,
                    ha='center',va='center',fontsize=28,zorder=7)
        ax.plot(-R,-R,'o',color='.35',ms=2.5,zorder=6)
        for endpoint in ((1.5,0),(0,1.5)):
            ax.add_patch(FancyArrowPatch((0,0),endpoint,arrowstyle='-|>',
                                        mutation_scale=12,lw=.85,color='black',zorder=7))
        ax.text(1.5,.16,r'$\bm e_{x,j}$',ha='center',va='bottom',fontsize=29)
        ax.text(.17,1.5,r'$\bm e_{y,j}$',ha='left',va='center',fontsize=29)
        # Only the SW quarter is the rounded erosion boundary at this corner.
        ax.add_patch(Arc((0,0),2*r,2*r,theta1=180,theta2=270,
                         edgecolor='#C93434',lw=.7,zorder=6))
        ax.plot(0,0,'o',color='black',ms=6,zorder=8)
        ax.annotate(r'$\bm p_j^{\mathrm{corn}}$',xy=(0,0),xytext=(-24,20),
                    textcoords='offset points',ha='right',va='bottom',fontsize=29,
                    arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3,rad=-.2',
                                    lw=1.1,color='black',mutation_scale=14,
                                    relpos=(0.42,0.0),patchA=None,shrinkA=2,shrinkB=5),zorder=12)
        ax.text(-3.2,1.6,r'$\mathcal C_j$',ha='center',va='center',fontsize=34)
        ax.text(1.2,-3.2,r'$\mathcal C_{j+1}$',ha='center',va='center',fontsize=34)
    else:
        ax.text(-3.15,-2.8,r'$\mathcal A_j$',ha='center',va='center',fontsize=36)
        ax.text(-3.2,1.6,r'$\mathcal C_j$',ha='center',va='center',fontsize=34)
        ax.text(1.1,-3.2,r'$\mathcal C_{j+1}$',ha='center',va='center',fontsize=34)
        ax.plot(0,0,'o',color='black',ms=5,zorder=10)
        ax.annotate(r'$\bm p_j^{\mathrm{corn}}$',(0,0),xytext=(24,-20),
                    textcoords='offset points',ha='left',va='top',fontsize=29,
                    arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3,rad=-.2',
                                    lw=1.1,color='black',mutation_scale=14,shrinkA=4,shrinkB=5),zorder=12)
    ax.set(xlim=(-4.8,2.2),ylim=(-4.5,2.2),aspect='equal')
    ax.set_axis_off()


def draw_example(ax):
    """Place a waypoint in A_j and draw its radius-R quarter-circle fillet."""
    waypoint = np.array([-1.2,-1.2])
    door = dict(x=(-WIDTH_X+r,-r),y=(-WIDTH_Y+r,-r))
    region = fillet_vertex_region(door,(),np.zeros(2),[0.,-1.],[1.,0.],r,R,
                                 pair_bounds=((-WIDTH_X,0.,-WIDTH_Y,6.),
                                              (-WIDTH_X,6.,-WIDTH_Y,0.)))
    if not region_contains_point(waypoint,region):
        raise RuntimeError('Illustrated waypoint is not in A_j.')
    center = waypoint+np.array([R,R])
    entry = waypoint+np.array([0.,R])
    exit = waypoint+np.array([R,0.])
    blue = 'tab:blue'
    # Nominal orthogonal polyline bends at p_j; the rounded path cuts it off.
    ax.plot([waypoint[0],waypoint[0],1.9],[1.9,waypoint[1],waypoint[1]],
            '--',color='0.5',lw=1,zorder=6)
    values = np.linspace(np.pi,1.5*np.pi,160)
    arc_points = center+R*np.array([np.cos(values),np.sin(values)]).T
    ax.plot(*arc_points.T,color=blue,lw=2.8,zorder=9)
    ax.plot([entry[0],entry[0]],[1.9,entry[1]],color=blue,lw=2.8,zorder=9)
    ax.plot([exit[0],1.9],[exit[1],exit[1]],color=blue,lw=2.8,zorder=9)
    for point in (entry,exit):
        ax.plot(*np.array([center,point]).T,':',color='0.45',lw=0.9,zorder=7)
        ax.plot(*point,'o',color=blue,ms=4,zorder=10)
    ax.plot(*center,'o',color='black',ms=4,zorder=10)
    ax.annotate(r'$\bm o_j$',center,xytext=(7,7),textcoords='offset points',
                fontsize=28,ha='left',va='bottom',zorder=12)
    ax.plot(*waypoint,'o',color='#884D70',ms=6,zorder=11)
    ax.annotate(r'$\bm p_j$',waypoint,xytext=(-7,-7),textcoords='offset points',
                fontsize=30,color='#884D70',ha='right',va='top',zorder=12)
    for first,last in (([-1.2,1.7],[-1.2,1.15]),([1.1,-1.2],[1.65,-1.2])):
        ax.add_patch(FancyArrowPatch(first,last,arrowstyle='-|>',color=blue,
                                    lw=1.5,mutation_scale=14,zorder=11))
    # Show the fillet radius R.
    ax.annotate(r'$R$',(center+entry)/2,xytext=(0,7),textcoords='offset points',
                fontsize=26,ha='center',va='bottom',zorder=12)


def create_figure():
    with plt.rc_context(FIGURE_STYLE):
        fig,axes = plt.subplots(1,2,figsize=(12.8,6.4))
        fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.94,wspace=.08)
        draw_region(axes[0],detailed=True)
        draw_region(axes[1],detailed=False)
        draw_example(axes[1])
        for ax,title in zip(axes,('a) Admissible waypoint region','b) Example feasible fillet')):
            ax.set_title(title,fontsize=24,pad=12)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    fig = create_figure()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(FIGURE_STYLE):
        for suffix in ('pdf','png'):
            path = args.output_dir/f'local_fillet_admissible_regions.{suffix}'
            fig.savefig(path,dpi=260,bbox_inches='tight',pad_inches=.04)
            print('Saved:',path)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
