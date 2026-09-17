"""Initial-heading ranges for the four combinations of turn direction.

Blue: no initial turn on the spot. Orange: an initial turn is required.
Sector boundaries and point-to-circle tangents are computed analytically.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Arc, Wedge, FancyArrowPatch, Patch
from tangent_convention_journal import STYLE

OUTPUT_DIRECTORY=Path(__file__).resolve().parent/'figures'
RADIUS=0.75
START=np.array([0.,0.])
CENTER=np.array([0.,2.70])
BLUE='#1686B0'
ORANGE='#D66B15'
RED='#F02C39'
GREY='0.55'
SYMBOL_FONT_SCALE=1.12
CONFIGURATIONS=[(1,1),(-1,-1),(1,-1),(-1,1)]


def legend_arrow(legend,orig_handle,xdescent,ydescent,width,height,fontsize):
    return FancyArrowPatch((-xdescent,height/2-ydescent),
                           (width-xdescent,height/2-ydescent),
                           arrowstyle='-|>',mutation_scale=16,
                           shrinkA=0,shrinkB=0)


def unit(angle):
    return np.array([np.cos(angle),np.sin(angle)])


def label(ax,point,symbol,color='black',size=14,ha='center'):
    ax.text(*point,symbol,fontsize=size*SYMBOL_FONT_SCALE,color=color,ha=ha,va='center',zorder=12)


def arrow(ax,start,end,color='black',lw=1.0,mutation_scale=10,**kwargs):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',color=color,lw=lw,
                 mutation_scale=mutation_scale,shrinkA=0,shrinkB=0,zorder=9,**kwargs))


def rotation(ax,center,radius,start,end,color='black'):
    values=np.linspace(start,end,100)
    points=center+radius*np.array([np.cos(values),np.sin(values)]).T
    ax.plot(*points.T,color=color,lw=1.0,zorder=7)
    arrow(ax,points[-5],points[-1],color)


def geometry(tau0,tau1):
    distance=np.linalg.norm(CENTER-START)
    if distance<=2*RADIUS:
        raise ValueError('This illustration requires the point to lie outside the 2R circle.')
    alpha=np.arctan2(*(CENTER-START)[::-1])
    beta=np.arcsin(RADIUS/distance)
    beta_prime=np.arcsin(2*RADIUS/distance)
    tangent_angle=alpha-tau1*beta
    half_range=np.pi/2-beta if tau0==tau1 else np.pi/2-(beta_prime-beta)
    boundary=alpha-tau1*np.pi/2 if tau0==tau1 else alpha-tau1*(beta_prime-np.pi/2)
    return alpha,beta,beta_prime,tangent_angle,half_range,boundary


def panel(ax,tau0,tau1,letter):
    alpha,beta,beta_prime,tangent_angle,half_range,boundary=geometry(tau0,tau1)
    equal=tau0==tau1
    sector_radius=0.70
    lo,hi=np.degrees([tangent_angle-half_range,tangent_angle+half_range])
    if letter=='a':
        lo,hi=0.0,np.degrees(tangent_angle)
    elif letter=='b':
        lo,hi=np.degrees(tangent_angle),180.0
    elif letter=='c':
        lo,hi=np.degrees([boundary,tangent_angle])
    elif letter=='d':
        lo,hi=np.degrees([tangent_angle,boundary])
    ax.add_patch(Wedge(START,sector_radius,hi,lo+360,facecolor=ORANGE,alpha=0.22,edgecolor='none'))
    ax.add_patch(Wedge(START,sector_radius,lo,hi,facecolor=BLUE,alpha=0.25,edgecolor='none'))
    ax.add_patch(Arc(START,2*sector_radius,2*sector_radius,theta1=hi,theta2=lo+360,color=ORANGE,lw=0.8))
    ax.add_patch(Arc(START,2*sector_radius,2*sector_radius,theta1=lo,theta2=hi,color=BLUE,lw=0.8))
    ax.add_patch(Circle(CENTER,RADIUS,fill=False,edgecolor=GREY,lw=0.9))
    if not equal:
        ax.add_patch(Circle(CENTER,2*RADIUS,fill=False,edgecolor=GREY,lw=0.9))
    ax.plot(*np.array([START,CENTER]).T,'--',color=GREY,lw=0.8)
    ax.plot(*START,'ko',ms=3,zorder=10)
    ax.plot(*CENTER,'ko',ms=3,zorder=10)
    leader_side=-tau1
    ax.annotate(r'$\mathbf{p}_1^o$',xy=CENTER,xytext=CENTER+[0.48*leader_side,-0.20],
                fontsize=15*SYMBOL_FONT_SCALE,ha='center',va='center',
                arrowprops=dict(arrowstyle='<|-',color='black',lw=0.65,
                                linestyle='--',shrinkA=0,shrinkB=3,
                                relpos=(1.0 if leader_side<0 else 0.0,0.5),
                                mutation_scale=9))
    circle_label=CENTER+[-0.53,0.28] if tau1==1 else CENTER+[0.53,0.28]
    label(ax,circle_label,r'$\mathcal{O}_1$',size=17)
    radial=alpha-0.9*tau1 if equal else alpha-0.5
    ax.plot(*np.array([CENTER,CENTER+RADIUS*unit(radial)]).T,color=GREY,lw=0.8)
    radius_label_offset=(0.18 if equal else 0.13)*np.sign(np.cos(radial))*unit(radial+np.pi/2)
    label(ax,CENTER+0.57*RADIUS*unit(radial)+radius_label_offset,r'$R$')
    if not equal:
        radial=2.12
        ax.plot(*np.array([CENTER,CENTER+2*RADIUS*unit(radial)]).T,color=GREY,lw=0.8)
        label(ax,CENTER+1.64*RADIUS*unit(radial)+[0.13,0.03],r'$2R$')
        label(ax,CENTER+[-1.27,0.38],r"$\mathcal{O}'_1$",size=17)
    tangent_colors = {
        'a': [(1, '#DA3036')],
        'b': [(1, '#9EBB22')],
        'c': [(1, '#9EBB22'), (2, '#FF812C')],
        'd': [(1, '#229D39'), (2, '#DA3036')],
    }
    for multiplier,color in tangent_colors[letter]:
        theta=alpha-tau1*np.arcsin(multiplier*RADIUS/np.linalg.norm(CENTER-START))
        contact=START+np.sqrt(np.sum((CENTER-START)**2)-(multiplier*RADIUS)**2)*unit(theta)
        ax.plot(*np.array([START,contact]).T,color=color,lw=1.0)
    angle_vertex=np.array([0.,1.60])
    ax.plot(*np.array([angle_vertex,angle_vertex+[0.30,0]]).T,color=GREY,lw=0.8)
    ax.add_patch(Arc(angle_vertex,0.42,0.42,theta1=0,theta2=90,color=GREY,lw=0.8))
    label(ax,angle_vertex+[0.26,0.20],r'$\alpha_0$')
    tangent_distance={'a':1.35,'b':1.10,'c':1.00,'d':1.00}[letter]
    tangent_vertex=START+tangent_distance*unit(tangent_angle)
    ax.plot(*np.array([tangent_vertex,tangent_vertex+[0.30,0]]).T,
            color=GREY,lw=0.8)
    ax.add_patch(Arc(tangent_vertex,0.42,0.42,theta1=0,
                     theta2=np.degrees(tangent_angle),color=GREY,lw=0.8))
    tangent_label=1.57*unit(tangent_angle)+[0.48*tau1,0]
    if letter=='b':
        tangent_label=tangent_vertex+[0.52,0.22]
    elif not equal:
        tangent_label=tangent_vertex+[0.52,0.20]
    label(ax,tangent_label,r'$\alpha_0-\tau_1\beta$',size=13)
    if not equal:
        outer_angle=alpha-tau1*beta_prime
        outer_vertex=START+1.65*unit(outer_angle)
        ax.plot(*np.array([outer_vertex,outer_vertex+[0.30,0]]).T,
                color=GREY,lw=0.8)
        ax.add_patch(Arc(outer_vertex,0.42,0.42,theta1=0,
                         theta2=np.degrees(outer_angle),color=GREY,lw=0.8))
        label(ax,outer_vertex+[0.62 if letter=='d' else 0.52,0.16 if letter=='d' else 0.26],r"$\alpha_0-\tau_1\beta'$",size=13)
    arrow(ax,START,1.05*unit(boundary),RED,lw=1.1,mutation_scale=16)
    end=1.12*unit(boundary)
    label(ax,end+[0,-0.16],r'$\theta_0$',RED,size=14)
    rotation(ax,START,0.23,np.pi/3,np.pi/3+tau0*1.55*np.pi)
    label(ax,np.array([0.,-0.32]),rf'$\tau_0={tau0:+d}$',size=14)
    rotation(ax,CENTER,RADIUS+0.13,-0.2 if tau1==1 else np.pi+0.2,
             1.05 if tau1==1 else np.pi-1.05)
    label(ax,CENTER+[0.96*tau1,0.75],rf'$\tau_1={tau1:+d}$',size=14)
    ax.annotate(r'$(x_0,y_0)$',xy=START,
                xytext=(-0.90 if tau0>0 else 0.90,0.29),
                fontsize=14*SYMBOL_FONT_SCALE,ha='center',va='center',zorder=12,
                arrowprops=dict(arrowstyle='<|-',color='black',lw=0.65,
                                linestyle='--',shrinkA=0,shrinkB=3,
                                relpos=(1.0 if tau0>0 else 0.0,0.5),
                                mutation_scale=9))
    ax.set_title(rf'{letter}) $\tau_0={tau0:+d},\ \tau_1={tau1:+d}$',fontsize=17,pad=0,y=0.97)
    ax.set(xlim=(-1.90,1.90),ylim=(-1.72,CENTER[1]+1.75),aspect='equal')
    ax.axis('off')


def create_figure():
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,4,figsize=(15,6.9))
        fig.subplots_adjust(left=0.025,right=0.975,bottom=0.08,top=0.93,wspace=0.05)
        for ax,(tau0,tau1),letter in zip(axes,CONFIGURATIONS,'abcd'):
            panel(ax,tau0,tau1,letter)
        divider_x=(axes[1].get_position().x1+axes[2].get_position().x0)/2
        fig.add_artist(Line2D([divider_x,divider_x],[0.09,0.93],
                              transform=fig.transFigure,color='0.75',lw=0.7))
        legend_center=(axes[0].get_position().x0+axes[1].get_position().x1)/2
        fig.legend(handles=[
            Patch(facecolor=BLUE,alpha=0.25,edgecolor='none',
                  label=r'$|\delta|\leq\pi/2-\beta$: {\fontsize{12.5}{15}\selectfont No initial turn}'),
            Patch(facecolor=ORANGE,alpha=0.22,edgecolor='none',
                  label=r'$|\delta|>\pi/2-\beta$: {\fontsize{12.5}{15}\selectfont Initial turn required}'),
            FancyArrowPatch((0,0),(1,0),arrowstyle='-|>',color=RED,lw=1.1,
                   label=r'$\theta_0=\alpha_0-\tau_1\pi/2$'),
        ],loc='center',bbox_to_anchor=(legend_center,0.17),
            fontsize=14,frameon=False,handlelength=1.5,labelspacing=0.7,
            labelcolor=[BLUE,ORANGE,RED],
            handler_map={FancyArrowPatch:HandlerPatch(patch_func=legend_arrow)})
        legend_center=(axes[2].get_position().x0+axes[3].get_position().x1)/2+0.02
        fig.legend(handles=[
            Patch(facecolor=BLUE,alpha=0.25,edgecolor='none',
                  label=r"$|\delta|\leq\pi/2-(\beta'-\beta)$: {\fontsize{12.5}{15}\selectfont No initial turn}"),
            Patch(facecolor=ORANGE,alpha=0.22,edgecolor='none',
                  label=r"$|\delta|>\pi/2-(\beta'-\beta)$: {\fontsize{12.5}{15}\selectfont Initial turn required}"),
            FancyArrowPatch((0,0),(1,0),arrowstyle='-|>',color=RED,lw=1.1,
                   label=r"$\theta_0=\alpha_0-\tau_1(\beta'-\pi/2)$"),
        ],loc='center',bbox_to_anchor=(legend_center,0.17),
            fontsize=14,frameon=False,handlelength=1.5,labelspacing=0.7,
            labelcolor=[BLUE,ORANGE,RED],
            handler_map={FancyArrowPatch:HandlerPatch(patch_func=legend_arrow)})
        return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true')
    args=parser.parse_args()
    figure=create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for ext in ('pdf','png'):
            output=OUTPUT_DIRECTORY/f'initial_heading_ranges.{ext}'
            figure.savefig(output,dpi=300,bbox_inches='tight',pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show: plt.close(figure)
    else: plt.show()


if __name__=='__main__':
    main()
