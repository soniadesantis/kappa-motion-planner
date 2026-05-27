from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr
from casadi import sin, cos

# Define problem parameters
x0, y0 = 0, 0
xf, yf = 0, 5

v_max, omega_max = 1, 1
v_min, omega_min = 0, -omega_max
R = v_max/ omega_max
r_wheel = 0.1
theta0 = 0 
thetaf = pi
r = 0.5 # Unicycle footprint
L = 0.5
omegaw_max = (v_max + 0.5*L*omega_max) / r_wheel
omegaw_min = -omegaw_max
# Create OCP
ocp = Ocp(T=FreeTime(10.0))

# Unicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

omegar = ocp.control()
omegal = ocp.control()

v = r_wheel *(omegar + omegal)/2
omega = r_wheel * (omegar - omegal)/L

ocp.set_der(x, v*cos(theta))
ocp.set_der(y, v*sin(theta))
ocp.set_der(theta, omega)

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==x0)
ocp.subject_to(ocp.at_t0(y)==y0)
ocp.subject_to(ocp.at_t0(theta) == theta0)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==xf)
ocp.subject_to(ocp.at_tf(y)==yf)
ocp.subject_to(ocp.at_tf(theta)==thetaf)


# ocp.subject_to(ocp.at_t0(cos(theta)) == cos(theta0))
# ocp.subject_to(ocp.at_t0(sin(theta)) == sin(theta0))

# ocp.subject_to(ocp.at_tf(cos(theta)) == cos(thetaf))
# ocp.subject_to(ocp.at_tf(sin(theta)) == sin(thetaf))

# ocp.subject_to(ocp.at_t0(cos(theta)) == cos(theta0))
# ocp.subject_to(ocp.at_t0(sin(theta)) == sin(theta0))

# ocp.subject_to(ocp.at_tf(cos(theta)) == cos(thetaf))
# ocp.subject_to(ocp.at_tf(sin(theta)) == sin(thetaf))

ocp.set_initial(x,x0)
ocp.set_initial(y,y0)
ocp.set_initial(theta, theta0)
ocp.set_initial(omegar, omegaw_max)
ocp.set_initial(omegal, omegaw_max)

# Wheel speed bounds (use wheel bounds here!)
ocp.subject_to(omegaw_min <= omegal)
ocp.subject_to(omegal <= omegaw_max)
ocp.subject_to(omegaw_min <= omegar)
ocp.subject_to(omegar <= omegaw_max)

# Minimal time
ocp.add_objective(ocp.T)

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=60,M=4,intg='rk'))

# solve
try:
    sol = ocp.solve()
    # self.ocp.show_infeasibilities(1e-5)
except:
    sol = ocp.non_converged_solution
    # self.ocp.show_infeasibilities(1e-5)

from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

dx0 = 0.3*cos(theta0)
dy0 = 0.3*sin(theta0)
dxf = 0.3*cos(thetaf)
dyf = 0.3*sin(thetaf)

quiver(x0, y0, dx0, dy0, angles='xy', scale_units='xy', scale=1)
quiver(xf, yf, dxf, dyf, angles='xy', scale_units='xy', scale=1)

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')

ts = np.linspace(0,2*pi,1000)

axis('equal')

figure()

ts, omegars = sol.sample(omegar, grid='control')
ts, omegals = sol.sample(omegal, grid='control')

plot(ts, omegars, 'r-', label='omegar')
plot(ts, omegals, 'b-', label='omegal')
legend()


show(block=True)