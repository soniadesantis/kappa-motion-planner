from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr

N = 60
# Define problem parameters
x0, y0 = 0, 0
xf, yf = 0, 10

v_max, omega_max = 1, 1
v_min, omega_min = 0, -omega_max

theta0 = -0.3
thetaf = pi + 0.3
r = 0.5 # Unicycle footprint

# Create OCP
ocp = Ocp(T=FreeTime(10.0))

# Unicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

v = ocp.control()
omega = ocp.control()

ocp.set_der(x, v*cos(theta))
ocp.set_der(y, v*sin(theta))
ocp.set_der(theta, omega)

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==x0)
ocp.subject_to(ocp.at_t0(y)==y0)
ocp.subject_to(ocp.at_t0(np.cos(theta)) == np.cos(theta0))

# Final constraint
ocp.subject_to(ocp.at_tf(x)==xf)
ocp.subject_to(ocp.at_tf(y)==yf)
ocp.subject_to(ocp.at_tf(np.cos(theta)) == np.cos(thetaf))

ocp.set_initial(x, np.linspace(x0, xf, N))
ocp.set_initial(y, np.linspace(y0, yf, N))
ocp.set_initial(theta, np.linspace(theta0, thetaf, N))
ocp.set_initial(v,v_max)
ocp.set_initial(omega, 0)

ocp.subject_to(v_min <= (v<=v_max))
ocp.subject_to(omega_min <= (omega<=omega_max))

# Minimal time
ocp.add_objective(ocp.T)

# Pick a solution method
options = { "expand": True,
            "verbose": False,
            "print_time": False, # True
            "error_on_fail": False,
            "ipopt": {	"linear_solver":"mumps", #"ma27", #
                        "print_level": 0, # 3, 5
                        "tol": 1e-6,
                        'sb': 'yes', # supress IPOPT banner
                        # 'hessian_approximation': 'limited-memory',
                        },
            "common_options":{"final_options" : {"dump_in":False, "dump_out":False, "dump_dir": "debug_run"}}
        }
# Pick a solution method
ocp.solver('ipopt', options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N,M=4,intg='rk'))

# solve
sol = ocp.solve()

T_sol = sol.value(ocp.T)
print(f"Optimal time: {T_sol:.6f} seconds")
from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')
dx0 = 0.3*cos(theta0)
dy0 = 0.3*sin(theta0)
dxf = 0.3*cos(thetaf)
dyf = 0.3*sin(thetaf)

quiver(x0, y0, dx0, dy0, angles='xy', scale_units='xy', scale=1)
quiver(xf, yf, dxf, dyf, angles='xy', scale_units='xy', scale=1)

ts = np.linspace(0,2*pi,1000)
axis('equal')
figure()

ts, vs = sol.sample(v, grid='control')
ts, omegas = sol.sample(omega, grid='control')

plot(ts, vs, 'r-', label='v')
plot(ts, omegas, 'b-', label='omega')
legend()


show(block=True)