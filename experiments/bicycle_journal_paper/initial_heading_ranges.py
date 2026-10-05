"""Minimum-time bicycle family selection for both prescribed target turns.

Four sectors combine selection of tau_0 with the CSC / C^bCSC thresholds.
Reference and family-switch headings are computed from alpha, beta and beta'.
Run with --no-show to save PDF and PNG without opening a window.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Wedge, Arc, FancyArrowPatch, Patch

from trajectory_constructions import STYLE

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'figures'
START = np.array([0., 0.])
CENTER = np.array([0., 2.70])
RADIUS = 0.75
DISK_RADIUS = 0.70
GREY = '0.55'
REFERENCE_COLOR = 'tab:red'
TANGENT_COLORS = {1: ('#9EBB22', '#FF812C'), -1: ('#229D39', '#DA3036')}
# Same relation colors in both panels; the sign of tau_0 changes with tau_1.
COLORS = {'CSC_equal': '#1686B0', 'CSC_opposite': '#229D39',
          'CbCSC_equal': '#D66B15', 'CbCSC_opposite': '#884D70'}
CONTOUR_COLORS = {'CSC_equal': '#0C6688', 'CSC_opposite': '#176B27',
                  'CbCSC_equal': '#A74B0C', 'CbCSC_opposite': '#60364F'}


def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrowPatch((-xdescent, height/2-ydescent),
                           (width-xdescent, height/2-ydescent),
                           arrowstyle='-|>', mutation_scale=16,
                           shrinkA=0, shrinkB=0)


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def label(ax, point, text, offset=(0, 0), color='black', size=22, **kwargs):
    return ax.annotate(text, point, xytext=offset, textcoords='offset points',
                       fontsize=size, color=color, zorder=20, **kwargs)


def arrow(ax, start, end, color='black', width=1.2, size=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=size, color=color, lw=width,
                                shrinkA=0, shrinkB=0, zorder=10))


def geometry(tau1):
    h = np.linalg.norm(CENTER-START)
    if h <= 3*RADIUS:
        raise ValueError('The selection illustration requires h > 3R.')
    alpha = np.arctan2(*(CENTER-START)[::-1])
    beta = np.arcsin(RADIUS/h)
    beta_prime = np.arcsin(2*RADIUS/h)
    reference = alpha-tau1*beta
    equal_limit = alpha-tau1*np.pi/2
    opposite_limit = alpha-tau1*(beta_prime-np.pi/2)
    return h, alpha, beta, beta_prime, reference, equal_limit, opposite_limit


def sectors(tau1):
    _, _, beta, beta_prime, reference, _, _ = geometry(tau1)
    equal_threshold = np.pi/2-beta
    opposite_threshold = np.pi/2-(beta_prime-beta)
    # phi = tau_1*(theta_0-theta_ref), modulo 2pi.
    intervals = [
        (0, opposite_threshold, 'CSC', -tau1, 'CSC_opposite'),
        (opposite_threshold, np.pi, 'CbCSC', -tau1, 'CbCSC_opposite'),
        (np.pi, 2*np.pi-equal_threshold, 'CbCSC', tau1, 'CbCSC_equal'),
        (2*np.pi-equal_threshold, 2*np.pi, 'CSC', tau1, 'CSC_equal'),
    ]
    return [(reference+tau1*low, reference+tau1*high, family, tau0, key)
            for low, high, family, tau0, key in intervals]


def draw_angle(ax, start, end, radius, text):
    low, high = sorted(np.degrees([start, end]))
    ax.add_patch(Arc(START, 2*radius, 2*radius, theta1=low, theta2=high,
                     color='0.3', lw=0.8, zorder=8))
    point = START+(radius+0.10)*unit((start+end)/2)
    label(ax, point, text, ha='center', va='center', size=14)


def panel(ax, tau1, letter):
    h, alpha, beta, beta_prime, reference, equal_limit, opposite_limit = geometry(tau1)
    for start, end, family, tau0, key in sectors(tau1):
        low, high = sorted(np.degrees([start, end]))
        ax.add_patch(Wedge(START, DISK_RADIUS, low, high,
                           facecolor=COLORS[key], alpha=0.25,
                           edgecolor='none', zorder=1))
        ax.add_patch(Arc(START, 2*DISK_RADIUS, 2*DISK_RADIUS,
                         theta1=low, theta2=high, color=CONTOUR_COLORS[key],
                         lw=0.9, zorder=4))

    for radius in (RADIUS, 2*RADIUS):
        ax.add_patch(Circle(CENTER, radius, fill=False, edgecolor=GREY,
                            lw=0.9, linestyle='-', zorder=2))
    ax.plot(*np.array([START, CENTER]).T, '--', color=GREY, lw=0.85, zorder=3)
    ax.plot(*CENTER, 'ko', ms=3, zorder=12)
    ax.plot(*START, 'ko', ms=3, zorder=12)
    label(ax, CENTER, r'$\mathbf o_1$', (-4*tau1, 4),
          ha='right' if tau1 > 0 else 'left', va='bottom')
    label(ax, START, r'$\mathbf p_0$', (0, -4), ha='center', va='top')
    label(ax, CENTER+np.array([-tau1*0.4, 0.40]), r'$\mathcal O_1$', ha='center')
    label(ax, CENTER+np.array([-tau1*1.06, 0.75]), r"$\mathcal O'_1$", ha='center', va='center')

    reference_color, auxiliary_color = TANGENT_COLORS[tau1]
    for radius, color in ((RADIUS, reference_color), (2*RADIUS, auxiliary_color)):
        theta = alpha-tau1*np.arcsin(radius/h)
        contact = START+np.sqrt(h*h-radius*radius)*unit(theta)
        ax.plot(*np.array([START, contact]).T, color=color, lw=1.0, zorder=7)
        ax.plot(*np.array([CENTER, contact]).T, ':', color=GREY, lw=0.8, zorder=3)
    for heading, key in ((equal_limit, 'CSC_equal'), (opposite_limit, 'CSC_opposite')):
        arrow(ax, START, START+1.05*unit(heading),
              color=CONTOUR_COLORS[key], width=1.1, size=16)
    # The two branch ties delimit the optimal initial-turn half-disks.
    ax.plot(*np.array([START, START+DISK_RADIUS*unit(reference+np.pi)]).T,
            linestyle=(0, (4, 2)), color='0.25', lw=1.6, zorder=9)
    # Measure theta_ref from the positive horizontal at a point on the tangent.
    vertex = START+1.40*unit(reference)
    ax.plot(*np.array([vertex, vertex+np.array([0.48, 0.])]).T,
            color=GREY, lw=0.9, zorder=8)
    ax.add_patch(Arc(vertex, 0.50, 0.50, theta1=0,
                     theta2=np.degrees(reference), color=GREY, lw=0.9, zorder=8))
    label(ax, vertex+0.48*unit(reference/2), r'$\theta_{\mathrm{ref}}$',
          ha='center', va='center', color=reference_color,
          bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

    # Prescribed direction on the target circle.
    turn_angles = np.linspace(-0.2 if tau1 > 0 else np.pi+0.2,
                              1.0 if tau1 > 0 else np.pi-1.0, 70)
    points = CENTER+(RADIUS+0.12)*np.array([np.cos(turn_angles), np.sin(turn_angles)]).T
    ax.plot(*points.T, color='black', lw=1, zorder=8)
    arrow(ax, points[-5], points[-1], size=8)
    ax.text(0.02, 0.98, rf'{letter}) $\tau_1={tau1:+d}$',
            transform=ax.transAxes, fontsize=26, ha='left', va='top')
    ax.set(xlim=(-2.05, 2.05), ylim=(-1.05, 4.70), aspect='equal')
    ax.axis('off')
    print(f'{letter}) tau1={tau1:+d}: reference={np.degrees(reference):.6f} deg, '
          f'equal switch={np.degrees(equal_limit):.6f} deg, '
          f'opposite switch={np.degrees(opposite_limit):.6f} deg')


def create_figure():
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(10, 7.5))
        figure.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.13, wspace=0.08)
        for ax, tau1, letter in zip(axes, (1, -1), 'ab'):
            panel(ax, tau1, letter)
        figure.canvas.draw()
        positions = [ax.get_position() for ax in axes]
        entries = [
            ('CSC_equal', r'$CSC$, $\tau_0=\tau_1$'),
            ('CSC_opposite', r'$CSC$, $\tau_0=-\tau_1$'),
            ('CbCSC_equal', r'$C^bCSC$, $\tau_0=\tau_1$'),
            ('CbCSC_opposite', r'$C^bCSC$, $\tau_0=-\tau_1$'),
        ]
        swatches = [Patch(facecolor=COLORS[key], alpha=0.25,
                          edgecolor='none', label=text) for key, text in entries]
        switch_arrows = [FancyArrowPatch(
            (0, 0), (1, 0), arrowstyle='-|>', color=CONTOUR_COLORS[key], lw=1.1,
            label=r'Switch between $CSC$ and $C^bCSC$'+'\n'+rf'$\tau_0={relation}\tau_1$')
            for key, relation in (('CSC_equal', ''), ('CSC_opposite', '-'))]
        # Preserve the existing two family rows and add a switch-arrow row.
        handles = [swatches[0], swatches[1], switch_arrows[0],
                   swatches[2], swatches[3], switch_arrows[1]]
        legend_colors = [CONTOUR_COLORS[key] for key in
                         ('CSC_equal', 'CSC_opposite', 'CSC_equal',
                          'CbCSC_equal', 'CbCSC_opposite', 'CSC_opposite')]
        figure.legend(handles=handles, ncol=2, frameon=False,
                      loc='upper center', bbox_to_anchor=(0.5, positions[0].y0-0.008),
                      fontsize=21, columnspacing=1.3, labelcolor=legend_colors,
                      handler_map={FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)})
        return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            output = OUTPUT_DIRECTORY / f'initial_heading_ranges.{extension}'
            figure.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.04)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
