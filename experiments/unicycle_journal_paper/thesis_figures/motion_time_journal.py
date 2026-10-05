"""Thesis version of the initial-turn selection time profiles.

Uses the journal example's geometry and analytical planner, with the notation
of Selection of the Initial Turn Direction. Run directly in VS Code, or pass
--no-show to export without opening a window. Journal files are untouched.
"""

import argparse
from math import asin, atan2, cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.helpers.pose_to_circle_unicycle import (
    compute_three_maneuvers_no_collision_avoidance,
)
from kappa_planner.trajectory import CurvilinearArcUnicycle
from kappa_planner.vehicle import Unicycle


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
STYLE = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.size': 23,
    'axes.labelsize': 25,
    'xtick.labelsize': 23,
    'legend.fontsize': 21,
}
START = np.array([0., 0.])
CENTER = np.array([0., 6.])
V_MAX = 2.
OMEGA_MAX = 1.
RADIUS = V_MAX / OMEGA_MAX
TAU_FINAL = 1
COLOR_EQUAL = '#008080'
COLOR_OPPOSITE = '#8E44AD'


def create_figure():
    """Compute the same profiles as the journal figure and restyle them."""
    alpha = atan2(*(CENTER - START)[::-1]) % (2 * pi)
    distance = np.linalg.norm(CENTER - START)
    beta = asin(RADIUS / distance)
    beta_prime = asin(2 * RADIUS / distance)
    theta_ref = alpha - TAU_FINAL * beta
    theta_end = theta_ref + 2 * pi
    target = CENTER + RADIUS * np.array([cos(alpha), sin(alpha)])
    target_heading = alpha + TAU_FINAL * pi / 2
    robot = Unicycle([*START, 0.], 0.430, length=0.430,
                     v_max=V_MAX, v_min=0., omega_max=OMEGA_MAX,
                     omega_min=-OMEGA_MAX)

    def traversal_time(theta, initial_turn):
        turn, arc, segment = compute_three_maneuvers_no_collision_avoidance(
            [*START, float(theta)], robot, *CENTER, TAU_FINAL,
            turn1=initial_turn,
        )
        final_arc = CurvilinearArcUnicycle(
            xc=CENTER[0], yc=CENTER[1],
            x0=segment.xf, y0=segment.yf, theta0=segment.theta,
            xf=target[0], yf=target[1], thetaf=target_heading,
            radius=RADIUS, turn_direction=TAU_FINAL,
            v=V_MAX, omega=OMEGA_MAX, unicycle=robot,
            t0=0., samples_number=5,
        )
        return sum(p.maneuver_time for p in (turn, arc, segment, final_arc))

    switch_equal = alpha + 3 * pi / 2
    switch_opposite = alpha - beta_prime + pi / 2
    time_equal = traversal_time(switch_equal, TAU_FINAL)
    time_opposite = traversal_time(switch_opposite, -TAU_FINAL)
    # Intersect the two linear TCSC branches (slopes -/+1/omega_max).
    slope = 1 / OMEGA_MAX
    theta_int = (slope * (switch_equal + switch_opposite)
                 + time_equal - time_opposite) / (2 * slope)
    time_int = time_equal - slope * (theta_int - switch_equal)
    np.testing.assert_allclose(
        [traversal_time(theta_int, TAU_FINAL),
         traversal_time(theta_int, -TAU_FINAL)], time_int, atol=1e-9,
    )
    time_ref = traversal_time(theta_ref, TAU_FINAL)
    np.testing.assert_allclose(
        traversal_time(theta_ref, -TAU_FINAL), time_ref, atol=1e-9,
    )
    # As in the journal plot, omit the duplicate periodic endpoint, where
    # the restricted-turn branches meet in the tangent-aligned configuration.
    headings = np.unique(np.r_[np.linspace(theta_ref, theta_end, 1000)[:-1],
                               switch_equal, switch_opposite, theta_int])
    equal = np.array([traversal_time(t, TAU_FINAL) for t in headings])
    opposite = np.array([traversal_time(t, -TAU_FINAL) for t in headings])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.plot(headings, equal, color=COLOR_EQUAL, lw=1.8,
                label=r'$\mathcal{T}^{\mathrm{eq}}$')
        ax.plot(headings, opposite, color=COLOR_OPPOSITE, lw=1.8,
                label=r'$\mathcal{T}^{\mathrm{neq}}$')
        ax.plot(switch_equal, time_equal, 'o', color=COLOR_EQUAL, ms=7,
                label=r'$\mathcal{T}^{\mathrm{eq}}$: $CSC\to TCSC$')
        ax.plot(switch_opposite, time_opposite, 'o', color=COLOR_OPPOSITE, ms=7,
                label=r'$\mathcal{T}^{\mathrm{neq}}$: $CSC\to TCSC$')
        ax.plot(theta_ref, time_ref, 'D', mfc='white', mec='black',
                mew=1., ms=8, zorder=5,
                label=r'Intersection at $\theta_{\mathrm{ref}}$')
        ax.plot(theta_int, time_int, 'o', color='black', ms=7, zorder=5,
                label=r'Intersection at $\theta_0^{\mathrm{int}}$')
        ticks = [theta_ref, switch_opposite, theta_int, switch_equal, theta_end]
        for tick in ticks:
            ax.axvline(tick, color='0.5', lw=0.7, ls='--', zorder=0)
        ax.set_xticks(ticks, [
            r'$\theta_{\mathrm{ref}}$',
            r'$\alpha-\beta^{\prime}+\frac{\pi}{2}$',
            r'$\theta_0^{\mathrm{int}}$',
            r'$\alpha+\frac{3\pi}{2}$',
            r'$\theta_{\mathrm{ref}}+2\pi$',
        ])
        ax.set_yticks([])
        ax.set_xlabel(r'Initial orientation $\theta_0$', labelpad=10)
        ax.set_ylabel(r'Total traversal time $\mathcal{T}$', labelpad=10)
        handles, labels = ax.get_legend_handles_labels()
        legend_order = [0, 2, 4, 1, 3, 5]
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.06),
                  handles=[handles[i] for i in legend_order],
                  labels=[labels[i] for i in legend_order],
                  ncol=2, frameon=False, columnspacing=1.1,
                  handlelength=1.1, handletextpad=0.4)
        ax.margins(x=0.035, y=0.055)
        fig.subplots_adjust(left=0.10, right=0.975,
                            bottom=1.104 / 5.4, top=1 - 1.632 / 5.4)
    print(f'theta_ref = {theta_ref:.9f} rad')
    print(f'theta_0^int = {theta_int:.9f} rad; T = {time_int:.9f} s')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    figure = create_figure()
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'png'):
            output = OUTPUT_DIRECTORY / f'motion_time_journal.{extension}'
            figure.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.06)
            print(f'Saved {output}')
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
