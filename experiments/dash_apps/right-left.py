import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from arena import wrapPositiveAngle, compute_extreme_poses_arc_line, compute_angular_difference_with_turn_direction, compute_mirr_circ
from math import sin, cos, pi, sqrt, atan2, asin, acos, tan
import numpy as np

#############################
## Initial values for the App
#############################

# Define tau1 and tau2 (they have to be different)
tau1 = -1 
tau2 = -tau1

# Define the maximum linear and angular velocities
v_max = 1
omega_max = 1
R = abs(v_max/omega_max)

# Define position of second circumference O2
xc2, yc2 = 0, 6

# Define the initial and final positions
x0, y0 = 0, 0
xt, yt, thetat = xc2 + R*cos(pi*0.5), yc2 + R*sin(pi*0.5), pi

# Define the number of elements in all the vectors
N = 10000

# Compute alpha0 (angle of the line connecting the center of O2 with (x0,y0))
alpha0 = atan2(yc2 - y0, xc2 - x0)

# Compute beta (angle between direction alpha0 and tangent to O2 passing through (x0,y0))
a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2) # a is the distance between (x0,y0) and the center of O2
beta = asin(R/a)

# Define the length of the segment that starts at (x0,y0) and ends at a point belonging to O2, tangent to O2
c = sqrt(a**2 - R**2)
# Define the length of the segment that starts at (x0,y0) and ends at a point belonging to O2_2r, tangent to O2_2r
c_2r = sqrt(a**2 - (2*R)**2)

# Compute beta_2R (angle between direction alpha0 and tangent to O2_2r passing through (x0,y0))
# O2_2r is the circumference with radius 2R and center in (xc2, yc2)
beta_2r = asin(2*R/a)

# Compute range of angles where turn on-the-spot is not needed theta0 in [theta_p, theta_pp]
theta_p = alpha0 - (tau1 + tau2)*0.5 * beta + (tau1 - tau2)*0.5 * beta
# Only valid for right-left
theta_pp = alpha0 + tau1*beta_2r + pi*0.5

# To build the theta0_vector, start from theta_p + off_set to avoid numerical problems
off_set = 0.01
theta0_min = theta_p + off_set
theta0_max = theta_p + 2*pi
theta0_vector = np.linspace(theta_p + off_set, theta_p + 2*pi, N, endpoint=False)

## Compute the quantities that are fixed for the case when tots is needed

# Compute delta0, as the direction between of the tangent to O2_2r passing through (x0,y0) 
delta0 = wrapPositiveAngle(alpha0 - tau2 * beta_2r)

# Define the first optimal circumference when tots is needed (this circumference is fixed)
xc1_star = x0 + R * cos(delta0)
yc1_star = y0 + R * sin(delta0)

# Compute the amplitude of the angle of the 2nd maneuver
iota2_star = tau1*pi*0.5

# Compute the optimal length of the segment when tots is needed
d3_star = sqrt((xc1_star - xc2)**2 + (yc1_star - yc2)**2 -4*R**2)

# Compute the amplitude of the angle of the 4th maneuver
iota4_star = compute_angular_difference_with_turn_direction(delta0, thetat, tau2)

# Define the mirrored circumference when tots is needed
xc2_mirr_star = xc2 + (2*R)*cos(delta0 - tau2*pi*0.5)
yc2_mirr_star = yc2 + (2*R)*sin(delta0 - tau2*pi*0.5)

# Compute the optimal length of the segment when tots is needed
d3_star_mirr = sqrt((xc1_star - xc2_mirr_star)**2 + (yc1_star - yc2_mirr_star)**2)

# Compute the extreme points of the segment when tots is needed (theta1_star corresponds to delta0)
x1_star, y1_star, theta1_star, x2_star, y2_star, _ = compute_extreme_poses_arc_line(xc1_star, yc1_star, xc2, yc2, tau1, tau2, R)

# Compute (xt,yt,thetat) mirrored when tots is needed
m = tan(delta0)
m_perp = -1/m
x_prime = (yt - m_perp*xt - y1_star + m*x1_star)/(m - m_perp)
y_prime = y1_star + m*(x_prime - x1_star)
xt_mirr = 2*x_prime - xt
yt_mirr = 2*y_prime - yt
thetat_mirr = 2*delta0 - thetat

# Compute the amplitude of the angle of the 4th maneuver mirrored
iota4_star_mirr = compute_angular_difference_with_turn_direction(delta0, thetat_mirr, -tau2)

## Initialize vectors
ind = 0
delta_vector = np.zeros(N)
delta0_vector = np.ones(N) * delta0
d2_vector = np.zeros(N)
d3_star_vector = np.ones(N) * d3_star
ineq_value_vector = np.zeros(N)
ineq_value_sympy_vector = np.zeros(N)
iota1_vector = np.zeros(N)
phi1_star_plus_iota2_star_vector = np.ones(N)
iota3_vector = np.zeros(N)
iota4_star_vector = np.ones(N) * iota4_star

total_time_3_man = np.zeros(N)
total_time_3_man_mirr = np.zeros(N)
total_time_4_man = np.zeros(N)
total_time_4_man_mirr = np.zeros(N)

tots_required = np.zeros(N, dtype = str)

for theta0 in theta0_vector:

    # Build the primitives without turn on-the-spot S_p = C_1S_1C_2
    xc1 = x0 + R * cos(theta0 + tau1 * pi*0.5)
    yc1 = y0 + R * sin(theta0 + tau1 * pi*0.5)
    x1, y1, theta1, x2, y2, theta1 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, tau1, tau2, R)
    iota1 = compute_angular_difference_with_turn_direction(theta0, theta1, tau1)
    iota3 = compute_angular_difference_with_turn_direction(theta1, thetat, tau2)
    delta = theta1
    d2 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    delta_vector[ind] = delta
    d2_vector[ind] = d2
    total_time_3_man[ind] = abs(iota1)/omega_max + d2/v_max + abs(iota3)/omega_max
    iota1_vector[ind] = abs(iota1)
    iota3_vector[ind] = abs(iota3)

    # Compute the maneuvers as if the circumference to reach is the mirrored one
    xc2_mirr_3man, yc2_mirr_3man, xt_mirr_3man, yt_mirr_3man, thetat_mirr_3man, total_time_mirr = compute_mirr_circ(xc1, yc1, xc2, yc2, R, delta, tau1, tau2, theta0, xt, yt, thetat, v_max, omega_max)
    x1_mirr, y1_mirr, theta1_mirr, x2_mirr, y2_mirr, _ = compute_extreme_poses_arc_line(xc1, yc1, xc2_mirr_3man, yc2_mirr_3man, tau1, -tau2, R)
    total_time_3_man_mirr[ind] = total_time_mirr

    # Build the primitives with turn on-the-spot S_p_star = T_1C_2S_3C_4 IN CASE NEEDED
    # Compute the angular difference between theta0 and gamma when rotating according to turn1 (needed for the check)
    ang_disp_to_check = compute_angular_difference_with_turn_direction(theta0, delta0, tau1)
    low_bound = compute_angular_difference_with_turn_direction(beta_2r, beta, tau1)
    up_bound = pi*0.5

    # If the tots is required
    if not(abs(low_bound) <= abs(ang_disp_to_check) <= up_bound) and not(abs(theta0 - theta_p) < 0.013): # and not(abs(theta0 - (theta_p + 2*pi)) < 0.013):
        phi1_star = compute_angular_difference_with_turn_direction(theta0, delta0 - tau1*pi*0.5, tau1)
        phi1_star_plus_iota2_star_vector[ind] = abs(phi1_star + iota2_star)
        iota4_star_vector[ind] = abs(iota4_star)
        total_time_4_man[ind] = abs(phi1_star)/omega_max + abs(iota2_star)/omega_max + d3_star/v_max + abs(iota4_star)/omega_max
        total_time_4_man_mirr[ind] = abs(phi1_star)/omega_max + abs(iota2_star)/omega_max + d3_star_mirr/v_max + abs(iota4_star_mirr)/omega_max
        ineq_value = (tau1 - tau2)*R*(delta - delta0) + d2 - d3_star
        string = 'Yes'
    else: 
        # Replace delta0 and d3 to show that when the tots is not needed, there values are the same as delta and d2
        delta0_vector[ind] = delta
        d3_star_vector[ind] = d2
        phi1_star_plus_iota2_star_vector[ind] = abs(iota1)
        iota4_star_vector[ind] = abs(iota3)
        total_time_4_man[ind] = total_time_3_man[ind]
        total_time_4_man_mirr[ind] = total_time_3_man_mirr[ind]
        ineq_value = 0
        string = 'No'

    # Need to check if this is needed...    
    ineq_value_sympy_vector[ind] = -sqrt(-12*sin(delta0) - 24*cos(delta0) + 41) + sqrt(-24*cos(delta) + 12*cos(theta0) - 4*cos(delta - theta0) + 41)
    + 12*cos(theta0) - 4*cos(delta - theta0)
    ineq_value_vector[ind] = ineq_value
    tots_required[ind] = string
    ind += 1

#############################
## Create a Dash app
#############################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB]) # Dash constructor, it initializes the Dash app

slider1 = html.Div([   
    dbc.Label("Theta0", html_for="size", class_name="text-center"),
    dcc.Slider(
    theta0_min,
    theta0_max,
    step= 0.01,
    marks = None, 
    value= theta0_min,
    tooltip={"placement": "bottom", "always_visible": True},
    id='theta0-slider'
    )])

theta0_options = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Theta0", className = "card-title"),
                slider1
            ]
        )
    ],
color = "light")

card = dbc.Card([
                dbc.CardBody([
                            html.H4("Useful variables", className="card-title"),
                            html.P(id="card-content", className="card-text"),
                            ])
                ])

graph = dbc.Card(
    [html.Div(id="plot1", className="text-danger"), dcc.Graph(id="plot-path")], style={"width": "800px", "height": "600px"} 
)

graph_delta = dbc.Card(
    [html.Div(id="plot2", className="text-danger"), dcc.Graph(id="plot-delta")]
)

second_graph = dbc.Card(
    [html.Div(id="plot1-1", className="text-danger"), dcc.Graph(id="plot-second_path")], style={"width": "800px", "height": "600px"} 
)

graph_d = dbc.Card(
    [html.Div(id="plot3", className="text-danger"), dcc.Graph(id="plot-d")]
)

graph_ineq = dbc.Card(
        [html.Div(id="plot4", className="text-danger"), dcc.Graph(id="plot-ineq")]
)

graph_motion_times = dbc.Card(
        [html.Div(id="plot5", className="text-danger"), dcc.Graph(id="plot-motion-times")]
)

graph_angles_amplitude = dbc.Card(
        [html.Div(id="plot6", className="text-danger"), dcc.Graph(id="plot-angles-amplitude")]
)
heading = html.H4(
    "Proof for right-left", className="bg-primary text-white p-2"
)


main_row = dbc.Row([
            # First Column with nested rows
            dbc.Col([
                # Nested Row 1
                dbc.Row([
                    dbc.Col([theta0_options])
                ], className="g-2",),
                # Nested Row 2 with two half-width columns
                dbc.Row([
                    dbc.Col([card])
                ], className="g-2",)
            ], width=3),  # Main Column taking half the width

            # dbc.Col([theta0_options], md=3),  # Main Column taking half the width
            # Second Column without nested rows
            # dbc.Col(graph, width=5),
            dbc.Col([
                dbc.Row([
                    dbc.Col(graph)
                ]),
                dbc.Row([
                    dbc.Col(second_graph)
                ]), 
                dbc.Row([
                    dbc.Col(graph_angles_amplitude)
                ])
            ], width = 5),
            dbc.Col([
                dbc.Row([
                    dbc.Col(graph_delta)
                ]),
                dbc.Row([
                    dbc.Col(graph_d)
                ]), 
                dbc.Row([
                    dbc.Col(graph_ineq)
                ]),
                dbc.Row([
                    dbc.Col(graph_motion_times)
                ])
            ], width = 4)
        ])

app.layout = html.Div([
    heading,
    dbc.Container(main_row, fluid = True)
])


@callback(
    Output('plot-path', 'figure'),
    Output('plot-second_path', 'figure'),
    Output('plot-delta', 'figure'),
    Output('plot-d', 'figure'),
    Output('plot-ineq', 'figure'),
    Output('plot-motion-times', 'figure'),
    Output('plot-angles-amplitude', 'figure'),
    Output('card-content', 'children'),
    Input('theta0-slider', 'value'))
def update_rectangle_plot(theta0):

    # Build the primitives without turn on-the-spot S_p = C_1S_2C_3
    xc1 = x0 + R * cos(theta0 + tau1 * pi*0.5)
    yc1 = y0 + R * sin(theta0 + tau1 * pi*0.5)
    x1, y1, theta1, x2, y2, theta1 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, tau1, tau2, R)
    iota1 = compute_angular_difference_with_turn_direction(theta0, theta1, tau1)
    iota3 = compute_angular_difference_with_turn_direction(theta1, thetat, tau2)
    theta1s = theta0
    theta1e = theta1s + iota1
    theta2s = theta1e
    theta2e = theta2s
    theta3s = theta2e
    theta3e = theta3s + iota3
    delta = theta1
    d2 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_time_3_man_point = abs(iota1)/omega_max + d2/v_max + abs(iota3)/omega_max

    # Compute the x and y coordinates of arc1 to plot it
    x_arc1 = xc1 + R * np.cos(np.linspace(theta1s - tau1 * pi*0.5, theta1e - tau1 * pi*0.5, 100))
    y_arc1 = yc1 + R * np.sin(np.linspace(theta1s - tau1 * pi/2, theta1e - tau1 * pi*0.5, 100))

    # Compute the x and y coordinates of arc3 to plot it
    x_arc3 = xc2 + R * np.cos(np.linspace(theta3s - tau2 * pi*0.5, theta3e - tau2 * pi*0.5, 100))
    y_arc3 = yc2 + R * np.sin(np.linspace(theta3s - tau2 * pi*0.5, theta3e - tau2 * pi*0.5, 100))

    # Build the primitives with the turn on-the-spot
    phi1_star = compute_angular_difference_with_turn_direction(theta0, delta0 - tau1*pi*0.5, tau1)
    theta1s_star = theta0
    theta1e_star = theta1s_star + phi1_star
    theta2s_star = theta1e_star
    theta2e_star = theta2s_star + iota2_star
    theta3s_star = theta2e_star
    theta3e_star = theta3s_star
    theta4s_star = theta3e_star
    theta4e_star = theta4s_star + iota4_star

    # Compute the x and y coordinates of arc2_star to plot it
    x_arc2_star = xc1_star + R * np.cos(np.linspace(theta2s_star - tau1 * pi*0.5, theta2e_star - tau1 * pi*0.5, 100))
    y_arc2_star = yc1_star + R * np.sin(np.linspace(theta2s_star - tau1 * pi*0.5, theta2e_star - tau1 * pi*0.5, 100))
    
    # Compute the x and y coordinates of arc4_star to plot it
    x_arc4_star = xc2 + R * np.cos(np.linspace(theta4s_star - tau2 * pi*0.5, theta4e_star - tau2 * pi*0.5, 100))
    y_arc4_star = yc2 + R * np.sin(np.linspace(theta4s_star - tau2 * pi*0.5, theta4e_star - tau2 * pi*0.5, 100))

    if not(abs(low_bound) <= abs(ang_disp_to_check) <= up_bound) and not(abs(theta0 - theta_p) < 0.013): 
        total_time_4_man_point = abs(phi1_star)/omega_max + abs(iota2_star)/omega_max + d3_star/v_max + abs(iota4_star)/omega_max
    else:
        total_time_4_man_point = total_time_3_man_point

    # Compute the maneuvers as if the circumference to reach is the mirrored one
    xc2_mirr_3man, yc2_mirr_3man, xt_mirr_3man, yt_mirr_3man, thetat_mirr_3man, total_time_3_man_point_mirr = compute_mirr_circ(xc1, yc1, xc2, yc2, R, delta, tau1, tau2, theta0, xt, yt, thetat, v_max, omega_max)
    
    # Update the strings
    delta0_string = f"""Delta0: {delta0:.6f}\nDelta: {delta:.6f}\nd2: {d2:.6f}\nd3_star: {d3_star:.6f} 
        \nInequality value: {ineq_value:.6f}\nIota4_star: {iota4_star:.6f}\nIota4_star_mirr: {iota4_star_mirr:.6f} 
        \nThetat: {thetat:.6f}\nThetat_mirr: {thetat_mirr:.6f}\nTotal time not mirrored: {total_time_3_man_point:.6f} 
        \nTotal time mirrored: {total_time_3_man_point_mirr:.6f}"""

    C1 ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc1 - R,  # Bottom-left corner x
        'y0': yc1 - R,  # Bottom-left corner y
        'x1': xc1 + R,  # Top-right corner x
        'y1': yc1 + R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7,
            'dash': 'dash'
        },
    }

    C2 ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc2 - R,  # Bottom-left corner x
        'y0': yc2 - R,  # Bottom-left corner y
        'x1': xc2 + R,  # Top-right corner x
        'y1': yc2 + R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7,
            'dash': 'dash'
        },
    }

    C1_star ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc1_star - R,  # Bottom-left corner x
        'y0': yc1_star - R,  # Bottom-left corner y
        'x1': xc1_star + R,  # Top-right corner x
        'y1': yc1_star + R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7,
            'dash': 'dash'
        },
    }

    C2_2R ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc2 - 2*R,  # Bottom-left corner x
        'y0': yc2 - 2*R,  # Bottom-left corner y
        'x1': xc2 + 2*R,  # Top-right corner x
        'y1': yc2 + 2*R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7, 
            'dash': 'dash'
        },
    }

    C2_mirr ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc2_mirr_star - R,  # Bottom-left corner x
        'y0': yc2_mirr_star - R,  # Bottom-left corner y
        'x1': xc2_mirr_star + R,  # Top-right corner x
        'y1': yc2_mirr_star + R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7, 
            'dash': 'dash'
        },
    }

    C2_mirr_3man ={
        'type': 'circle',
        'xref': 'x', 'yref': 'y',
        'x0': xc2_mirr_3man - R,  # Bottom-left corner x
        'y0': yc2_mirr_3man - R,  # Bottom-left corner y
        'x1': xc2_mirr_3man + R,  # Top-right corner x
        'y1': yc2_mirr_3man + R,  # Top-right corner y
        'line': {
            'color': 'black',
            'width': 0.7, 
            'dash': 'dash'
        },
    }

    # Define initial arrow
    arrow_length = 0.3
    initial_arrow = dict(
                    x= x0 + arrow_length * cos(theta0), y= y0 + arrow_length * sin(theta0),
                    xref="x", yref="y",
                    text="",
                    showarrow=True,
                    axref="x", ayref='y',
                    ax=x0 , ay=y0,
                    arrowhead=3,
                    arrowwidth=2,
                    arrowcolor='rgb(255,51,0)',)
    
    final_arrow = dict(
                x=xt + arrow_length * cos(thetat), y=yt + arrow_length * sin(thetat),
                xref="x", yref="y",
                text="",
                showarrow=True,
                axref="x", ayref='y',
                ax=xt , ay=yt,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor='rgb(255,51,0)',)
    
    final_arrow_mirr = dict(
                x=xt_mirr + arrow_length * cos(thetat_mirr), y=yt_mirr+ arrow_length * sin(thetat_mirr),
                xref="x", yref="y",
                text="",
                showarrow=True,
                axref="x", ayref='y',
                ax=xt_mirr , ay=yt_mirr,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor='rgb(255,51,0)',)
    
    final_arrow_mirr_3man = dict(
                x=xt_mirr_3man + arrow_length * cos(thetat_mirr_3man), y=yt_mirr_3man+ arrow_length * sin(thetat_mirr_3man),
                xref="x", yref="y",
                text="",
                showarrow=True,
                axref="x", ayref='y',
                ax=xt_mirr_3man , ay=yt_mirr_3man,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor='rgb(255,51,0)',)
    
    figure1 = {
        'data': [
            go.Scatter(
                x=[x0, x0 + c*cos(alpha0 - tau2*beta)], y=[y0, y0+c*sin(alpha0 - tau2*beta)], mode='lines',
                line=dict(color='black', width=0.7),
                name='Tangent 1'
            ),
            go.Scatter(
                x=[x0, x0 + c_2r*cos(alpha0 - tau2*beta_2r)], y=[y0, y0 + c_2r*sin(alpha0 - tau2*beta_2r)], mode='lines',
                line=dict(color='black', width=0.7),
                name='Tangent 2'
            ),
        # Add the point at (xt_mirr, yt_mirr)
        go.Scatter(
            x=[x0, xt_mirr], y=[y0, yt_mirr], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc, yc)
        go.Scatter(
            x=[x0], y=[y0], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc1, yc1)
        go.Scatter(
            x=[xc1], y=[yc1], mode='markers',
            marker=dict(color='black', size=5, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc1_star, yc1_star)
        go.Scatter(
            x=[xc1_star], y=[yc1_star], mode='markers',
            marker=dict(color='black', size=5, symbol='circle'),
            name='Point'
        ),

        # Add the optimal path
        go.Scatter(
            x=x_arc2_star, y=y_arc2_star, mode='lines',
            line=dict(color='blue', width=2),
            name='Arc2*'
            ),
        go.Scatter(
            x=x_arc4_star, y=y_arc4_star, mode='lines',
            line=dict(color='blue', width=2),
            name='Arc4*'
            ), 
        go.Scatter(
            x=[x1_star, x2_star], y=[y1_star, y2_star], mode='lines',
            line=dict(color='blue', width=2),
            name='Segment 3*'
            ),
        # Add the path
        go.Scatter(
            x=x_arc1, y=y_arc1, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Arc1'
            ),
        go.Scatter(
            x=x_arc3, y=y_arc3, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Arc3'
            ), 
        go.Scatter(
            x=[x1, x2], y=[y1, y2], mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Segment 2'
            ),
        ],
        'layout': go.Layout(
            title=f'Path',
            shapes = [C1, C2, C1_star, C2_2R, C2_mirr],
            annotations = [initial_arrow, final_arrow, final_arrow_mirr],
            hovermode='closest',
            showlegend=False,
            autosize=False,
            width= 600,  # Set desired width
            height=600 # Set desired height
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    figure11 = {
        'data': [
            go.Scatter(
                x=[x0, x0 + c*cos(alpha0 - tau2*beta)], y=[y0, y0+c*sin(alpha0 - tau2*beta)], mode='lines',
                line=dict(color='black', width=0.7),
                name='Tangent 1'
            ),
            go.Scatter(
                x=[x0, x0 + c_2r*cos(alpha0 - tau2*beta_2r)], y=[y0, y0 + c_2r*sin(alpha0 - tau2*beta_2r)], mode='lines',
                line=dict(color='black', width=0.7),
                name='Tangent 2'
            ),
        # Add the point at (xt_mirr, yt_mirr)
        go.Scatter(
            x=[x0, xt_mirr], y=[y0, yt_mirr], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc, yc)
        go.Scatter(
            x=[x0], y=[y0], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc1, yc1)
        go.Scatter(
            x=[xc1], y=[yc1], mode='markers',
            marker=dict(color='black', size=5, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xc1_star, yc1_star)
        go.Scatter(
            x=[xc1_star], y=[yc1_star], mode='markers',
            marker=dict(color='black', size=5, symbol='circle'),
            name='Point'
        ),
        # Add the point at (xt_mirr_3man, yt_mirr_3man)
        go.Scatter(
            x=[xt_mirr_3man], y=[yt_mirr_3man], mode='markers',
            marker=dict(color='black', size=5, symbol='circle'),
            name='Point'
        ),

        # Add the optimal path
        # go.Scatter(
        #     x=x_arc2_star, y=y_arc2_star, mode='lines',
        #     line=dict(color='blue', width=2),
        #     name='Arc2*'
        #     ),
        # go.Scatter(
        #     x=x_arc4_star, y=y_arc4_star, mode='lines',
        #     line=dict(color='blue', width=2),
        #     name='Arc4*'
        #     ), 
        # go.Scatter(
        #     x=[x1_star, x2_star], y=[y1_star, y2_star], mode='lines',
        #     line=dict(color='blue', width=2),
        #     name='Segment 3*'
        #     ),
        # Add the path
        go.Scatter(
            x=x_arc1, y=y_arc1, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Arc1'
            ),
        go.Scatter(
            x=x_arc3, y=y_arc3, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Arc3'
            ), 
        go.Scatter(
            x=[x1, x2], y=[y1, y2], mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Segment 2'
            ),
        ],
        'layout': go.Layout(
            title=f'Path',
            shapes = [C1, C2, C1_star, C2_2R, C2_mirr, C2_mirr_3man],
            annotations = [initial_arrow, final_arrow, final_arrow_mirr, final_arrow_mirr_3man],
            hovermode='closest',
            showlegend=False,
            autosize=False,
            width= 600,  # Set desired width
            height=600 # Set desired height
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }
    # Create the figure 
    figure2 = {
        'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=delta_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Angle Delta'
            ),
        go.Scatter(
            x=theta0_vector, y=delta0_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='Angle Delta0'
            ),
        go.Scatter(
            x=[theta0], y=[delta], mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Delta'
            )
        ],
        'layout': go.Layout(
            title=f'Angle Delta',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    # Create the figure 
    figure3 = {
    'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=d2_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Distance d2'
        ),
        go.Scatter(
            x=theta0_vector, y=d3_star_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='Distance d3_star'
        ),
        go.Scatter(
            x=[theta0], y=[d2], mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='d2'
            )
    ],
    'layout': go.Layout(
        title='Distance d2',
        shapes=[],  # You can add shapes here if needed
        hovermode='closest',
        showlegend=False,
        autosize=True
        # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    # Create the figure 
    figure4 = {
        'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=ineq_value_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Inequality value'
            ),
        # Add the inequality value computed with sympy
        go.Scatter(
            x=theta0_vector, y=ineq_value_sympy_vector, 
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Inequality value'
            ),
        go.Scatter(
            x=theta0_vector, y=np.zeros(N), 
            mode='lines',
            line=dict(color='red', width=1),
            name='Zero horizontal line'
            )
        ],
        'layout': go.Layout(
            title=f'Inequality value: (tau1-tau2)*R*(delta - delta0) + d2 - d3_star',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    # Create the figure 
    figure5 = {
        'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=total_time_3_man, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Total time w/o tots'
            ),
        # Add the inequality value computed with sympy
        go.Scatter(
            x=theta0_vector, y=total_time_3_man_mirr, 
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Total time w/o tots mirrored'
            ),
        go.Scatter(
            x=theta0_vector, y=total_time_4_man, 
            mode='lines',
            line=dict(color='blue', width=1),
            name='Total time with tots'
            ),
        go.Scatter(
            x=theta0_vector, y=total_time_4_man_mirr, 
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            name='Total time with tots mirrored'
            ),
        go.Scatter(
            x=[theta0], y=[total_time_3_man_point], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        go.Scatter(
            x=[theta0], y=[total_time_4_man_point], mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Point'
        ),
        ],
        'layout': go.Layout(
            title=f'Inequality value: (tau1-tau2)*R*(delta - delta0) + d2 - d3_star',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    # Create the figure 
    figure6 = {
        'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=iota1_vector, 
            mode='lines',
            line=dict(color='green', width=1),
            name='iota1'
            ),
        # Add the inequality value computed with sympy
        go.Scatter(
            x=theta0_vector, y=phi1_star_plus_iota2_star_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='phi1_star + iota2_star'
            ),
        go.Scatter(
            x=theta0_vector, y=abs(iota2_star) * np.ones(N), 
            mode='lines',
            line=dict(color='red', width=1),
            name='iota2_star'
            ),
        go.Scatter(
            x=theta0_vector, y=iota4_star_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='iota4_star'
            ),
        go.Scatter(
            x=theta0_vector, y=iota3_vector, 
            mode='lines',
            line=dict(color='green', width=1),
            name='iota3'
            ),
        # go.Scatter(
        #     x=[theta0], y=[total_time_3_man_point], mode='markers',
        #     marker=dict(color='red', size=10, symbol='circle'),
        #     name='Point'
        # ),
        # go.Scatter(
        #     x=[theta0], y=[total_time_4_man_point], mode='markers',
        #     marker=dict(color='red', size=10, symbol='circle'),
        #     name='Point'
        # ),
        ],
        'layout': go.Layout(
            title=f'Plot of iota1 and phi1_star + iota2_star',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }


    # Create the figure 
    figure7 = {
        'data': [
        # Add the path
        go.Scatter(
            x=theta0_vector, y=iota1_vector, 
            mode='lines',
            line=dict(color='green', width=1),
            name='iota1'
            ),
        # Add the inequality value computed with sympy
        go.Scatter(
            x=theta0_vector, y=phi1_star_plus_iota2_star_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='phi1_star + iota2_star'
            ),
        go.Scatter(
            x=theta0_vector, y=abs(iota2_star) * np.ones(N), 
            mode='lines',
            line=dict(color='red', width=1),
            name='iota2_star'
            ),
        go.Scatter(
            x=theta0_vector, y=iota4_star_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='iota4_star'
            ),
        go.Scatter(
            x=theta0_vector, y=iota3_vector, 
            mode='lines',
            line=dict(color='green', width=1),
            name='iota3'
            ),
        # go.Scatter(
        #     x=[theta0], y=[total_time_3_man_point], mode='markers',
        #     marker=dict(color='red', size=10, symbol='circle'),
        #     name='Point'
        # ),
        # go.Scatter(
        #     x=[theta0], y=[total_time_4_man_point], mode='markers',
        #     marker=dict(color='red', size=10, symbol='circle'),
        #     name='Point'
        # ),
        ],
        'layout': go.Layout(
            title=f'Plot of iota1 and phi1_star + iota2_star',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    return figure1, figure11, figure3, figure2, figure4, figure5, figure6, delta0_string




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)