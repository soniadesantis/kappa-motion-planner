from math import atan2, sin, cos, pi

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.geometry import Pose, Point, Circle
from kappa_planner.helpers.geometry_operations import select_tangency_point_from_point_circle
from kappa_planner.helpers.primitives import compute_extreme_poses_arc_line
from kappa_planner.trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle
from kappa_planner.vehicle import Unicycle

if __name__ == "__main__":

    ### Define Unicycle vehicle ###
    vehicle_width = 0.430
    vehicle_length = 0.430
    vehicle_vmax = 2
    vehicle_omegamax = 1

    unicycle = Unicycle(
    state = [0,0,0],
    width = vehicle_width,
    length = vehicle_length,
    v_max = vehicle_vmax,
    v_min = -vehicle_vmax,
    omega_max = vehicle_omegamax,
    omega_min = -vehicle_omegamax)

    x0, y0 = 0, 0
    xf, yf = 0, 10
    R = 2
    tau0 = 1
    tauf = -1

    circle_at_final_point = Circle(Point(xf, yf), 2 * R)

    xf_ref, yf_ref = select_tangency_point_from_point_circle(
        Point(x0, y0),
        circle_at_final_point,
        turn_direction=tauf,
        tol=1e-9,
    )

    alpha0 = atan2(yf_ref - y0, xf_ref - x0)
    alphaf = atan2(y0 - yf_ref, x0 - xf_ref)

    theta0 = alpha0 - tau0 * np.pi / 2
    thetaf = alphaf - tauf * np.pi / 2

    xc0, yc0 = x0 + R * cos(alpha0), y0 + R * sin(alpha0)
    xcf, ycf = xf + R * cos(alphaf), yf + R * sin(alphaf)

    circle0 = Circle(Point(xc0, yc0), R)
    circlef = Circle(Point(xcf, ycf), R)

    (
    x1,
    y1,
    theta1,
    x2,
    y2,
    theta2
    ) = compute_extreme_poses_arc_line(
        xc0,
        yc0,
        xcf,
        ycf,
        tau0,
        tauf,
        R,
        overlap=False
    )


    arc2 = CurvilinearArcUnicycle(
        xc=xc0,
        yc=yc0,
        x0=x0,
        y0=y0,
        theta0=theta0,
        xf=x1,
        yf=y1,
        thetaf=theta1,
        radius=R,
        turn_direction=tau0,
        v=R,
        omega=1,
        unicycle= unicycle,
        t0=0,
        samples_number=10,
    )

    arc4 = CurvilinearArcUnicycle(
        xc=xcf,
        yc=ycf,
        x0=x2,
        y0=y2,
        theta0=theta2,
        xf=xf,
        yf=yf,
        thetaf=thetaf,
        radius=R,
        turn_direction=tauf,
        v=R,
        omega=1,
        unicycle= unicycle,
        t0=0,
        samples_number=10,
    )

    print(f"Arc amplitdue: {arc2.iota:.4f} rad, {arc2.iota * 180/pi:.4f} deg.")
    print(f"Arc amplitdue: {arc4.iota:.4f} rad, {arc4.iota * 180/pi:.4f} deg.")
    plt.figure()
    angle_array = np.linspace(0, 2 * np.pi, 100)
    plt.plot(
        circle0.center.x + circle0.radius * np.cos(angle_array),
        circle0.center.y + circle0.radius * np.sin(angle_array),
        "r--",
        label="Circle at start",
    )
    plt.plot(
        circlef.center.x + circlef.radius * np.cos(angle_array),
        circlef.center.y + circlef.radius * np.sin(angle_array),
        "g--",
        label="Circle at goal",
    )

    plt.plot(
        circle_at_final_point.center.x + circle_at_final_point.radius * np.cos(angle_array),
        circle_at_final_point.center.y + circle_at_final_point.radius * np.sin(angle_array),
        "m--",
        label="Circle at final point",
    )

    plt.plot(x0, y0, "ro", label="Start")
    plt.arrow(x0, y0, 0.5 * cos(theta0), 0.5 * sin(theta0), head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.plot(xf, yf, "go", label="Goal")
    plt.arrow(xf, yf, 0.5 * cos(thetaf), 0.5 * sin(thetaf), head_width=0.1, head_length=0.1, fc='g', ec='g')
    plt.plot(xc0, yc0, "rx", label="Circle at start")
    plt.plot(xcf, ycf, "gx", label="Circle at goal")
    plt.plot(xf_ref, yf_ref, "bx", label="Tangency point")
    plt.plot([x1, x2], [y1, y2], "r-", label="Arc from start")
    plt.legend()
    plt.axis("equal")
    plt.title("Pose-to-Pose Unicycle Problem")
    plt.show(block = True)

