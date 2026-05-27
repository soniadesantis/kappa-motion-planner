import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from arena import wrapPositiveAngle, compute_extreme_poses_arc_line, compute_angular_difference_with_turn_direction, compute_mirr_circ
from math import sin, cos, pi, sqrt, atan2, asin, acos, tan
from ocp_function import ocp_function
from multi_stage_ocp_function import ms_ocp_function
import numpy as np

#############################
## Initial values for the App
#############################

# Define tau1 and tau2 (they have to be different)
tau1 = 1 
tau2 = tau1

# Define the maximum linear and angular velocities
v_max = 1
v_min = 0
omega_max = 1
omega_min = -omega_max
R = abs(v_max/omega_max)

# Define position of second circumference O2
xc2, yc2 = 0, 6

# Define the initial and final positions
x0, y0 = 0, 0
xt, yt, thetat = xc2 + R*cos(pi*0.5), yc2 + R*sin(pi*0.5), 3*pi

# Define the number of elements in all the vectors
N = 10000

# Compute alpha0 (angle of the line connecting the center of O2 with (x0,y0))
alpha0 = atan2(yc2 - y0, xc2 - x0)

# Compute beta (angle between direction alpha0 and tangent to O2 passing through (x0,y0))
a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2) # a is the distance between (x0,y0) and the center of O2
beta = asin(R/a)

# Define the length of the segment that starts at (x0,y0) and ends at a point belonging to O2, tangent to O2
c = sqrt(a**2 - R**2)

# Compute range of angles where turn on-the-spot is not needed theta0 in [theta_p, theta_pp]
theta_p = alpha0 - (tau1 + tau2)*0.5 * beta + (tau1 - tau2)*0.5 * beta # alpha0 - beta (left-left)
# Only valid for right-left
theta_pp = alpha0 + 3* pi*0.5 # alpha0 + 270deg

# To build the theta0_vector, start from theta_p + off_set to avoid numerical problems
off_set = 0.01
theta0_min = theta_p + off_set
theta0_max = theta_p + 2*pi
theta0_vector = np.linspace(theta_p + off_set, theta_p + 2*pi, N, endpoint=False)

## Compute the quantities that are fixed for the case when tots is needed

# Define the first optimal circumference when tots is needed (this circumference is fixed)
xc1_star = x0 + R * cos(alpha0)
yc1_star = y0 + R * sin(alpha0)

# Compute the amplitude of the angle of the 2nd maneuver
iota2_star = tau1*pi*0.5

# Compute the optimal length of the segment when tots is needed
d3_star = sqrt((xc1_star - xc2)**2 + (yc1_star - yc2)**2)

# Compute the amplitude of the angle of the 4th maneuver
iota4_star = compute_angular_difference_with_turn_direction(alpha0, thetat, tau2)

# Compute the extreme points of the segment when tots is needed (theta1_star corresponds to delta0)
x1_star, y1_star, theta1_star, x2_star, y2_star, _ = compute_extreme_poses_arc_line(xc1_star, yc1_star, xc2, yc2, tau1, tau2, R)

## Initialize vectors
ind = 0
alpha_vector = np.zeros(N)
alpha0_vector = np.ones(N) * alpha0
d2_vector = np.zeros(N)
d3_star_vector = np.ones(N) * d3_star
ineq_value_vector = np.zeros(N)
iota1_vector = np.zeros(N)
phi1_star_plus_iota2_star_vector = np.ones(N)
iota3_vector = np.zeros(N)
iota4_star_vector = np.ones(N) * iota4_star
epsilon_vector = np.zeros(N)
phi1_star_minus_tau1 = np.zeros(N)
total_time_4_man_minus_tau1 = np.zeros(N)
expr_to_check = np.zeros(N)
sum_angles_3_man = np.zeros(N)
sum_angles_4_man = np.zeros(N)
total_ang_disp_vector = np.zeros(N)

total_time_3_man = np.zeros(N)
total_time_4_man = np.zeros(N)

tots_required = np.zeros(N, dtype = str)

for theta0 in theta0_vector:
    total_ang_disp_vector[ind] = tau1*(thetat-theta0)
    # Build the primitives without turn on-the-spot S_p = C_1S_1C_2
    xc1 = x0 + R * cos(theta0 + tau1 * pi*0.5)
    yc1 = y0 + R * sin(theta0 + tau1 * pi*0.5)
    x1, y1, theta1, x2, y2, theta1 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, tau1, tau2, R)
    iota1 = compute_angular_difference_with_turn_direction(theta0, theta1, tau1)
    iota3 = compute_angular_difference_with_turn_direction(theta1, thetat, tau2)
    alpha = theta1
    d2 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    alpha_vector[ind] = alpha
    d2_vector[ind] = d2
    total_time_3_man[ind] = abs(iota1)/omega_max + d2/v_max + abs(iota3)/omega_max
    iota1_vector[ind] = abs(iota1)
    iota3_vector[ind] = abs(iota3)

    # Build the primitives with turn on-the-spot S_p_star = T_1C_2S_3C_4 IN CASE NEEDED
    # Compute the angular difference between theta0 and gamma when rotating according to turn1 (needed for the check)
    ang_disp_to_check = compute_angular_difference_with_turn_direction(theta0, alpha0, tau1)
    low_bound = beta
    up_bound = pi*0.5

    # If the tots is required
    if not(abs(low_bound) <= abs(ang_disp_to_check) <= up_bound) and not(abs(theta0 - theta_p) < 0.013): # and not(abs(theta0 - (theta_p + 2*pi)) < 0.013):
        phi1_star = compute_angular_difference_with_turn_direction(theta0, alpha0 - tau1*pi*0.5, tau1)
        epsilon_vector[ind] = abs(phi1_star)
        phi1_star_plus_iota2_star_vector[ind] = abs(phi1_star + iota2_star)
        iota4_star_vector[ind] = abs(iota4_star)
        total_time_4_man[ind] = abs(phi1_star)/omega_max + abs(iota2_star)/omega_max + d3_star/v_max + abs(iota4_star)/omega_max
        total_time_4_man_minus_tau1[ind] = total_time_4_man[ind] 
        ineq_value =  d2 - d3_star
        string = 'Yes'
    else: 

        # Replace delta0 and d3 to show that when the tots is not needed, there values are the same as delta and d2

        phi1_star_minus_tau1[ind] = compute_angular_difference_with_turn_direction(theta0, alpha0 - tau1*pi*0.5, -tau1)
        sum_angles_3_man[ind] = abs(iota1) + abs(iota3)
        sum_angles_4_man[ind] = abs(phi1_star_minus_tau1[ind]) + abs(iota2_star) + abs(iota4_star)
        epsilon_vector[ind] = abs(phi1_star_minus_tau1[ind])
        total_time_4_man_minus_tau1[ind] = abs(phi1_star_minus_tau1[ind])/omega_max + abs(iota2_star)/omega_max + d3_star/v_max + abs(iota4_star)/omega_max
        alpha0_vector[ind] = alpha
        d3_star_vector[ind] = d3_star
        phi1_star_plus_iota2_star_vector[ind] = abs(iota1)
        iota4_star_vector[ind] = abs(iota3)
        total_time_4_man[ind] = total_time_3_man[ind]
        ineq_value = 0
        string = 'No'
    
    expr_to_check[ind] = 2*R*(R+d3_star) *(1 - cos(epsilon_vector[ind]))
    ind += 1
# print(phi1_star_plus_iota2_star_vector)
# print("\n\n")
# print(iota1_vector)
# print("\n\n")	
# print(phi1_star_plus_iota2_star_vector - iota1_vector)
# print("\n\n")
# print(iota4_star_vector - iota3_vector)

# print("\n\n")
# print(abs(phi1_star_plus_iota2_star_vector - iota1_vector) - abs(iota4_star_vector - iota3_vector) <= 0.000001)
# print("\n\n")
# print(abs(phi1_star_plus_iota2_star_vector + iota4_star_vector) - abs(iota1_vector + iota3_vector) <= 0.000001)
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

graph_forward_velocoty = dbc.Card(
        [html.Div(id="plot01", className="text-danger"), dcc.Graph(id="plot-forward-velocity")]
)

graph_angular_velocoty = dbc.Card(
        [html.Div(id="plot02", className="text-danger"), dcc.Graph(id="plot-angular_velocity")]
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
                ], className="g-2",),
                dbc.Row([
                    dbc.Col([graph_forward_velocoty])
                ]),
                # Nested Row 2 with two half-width columns
                dbc.Row([
                    dbc.Col([graph_angular_velocoty])
                ])
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
    Output('plot-forward-velocity', 'figure'),
    Output('plot-angular_velocity', 'figure'),
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
    alpha = theta1
    d2 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_time_3_man_point = abs(iota1)/omega_max + d2/v_max + abs(iota3)/omega_max

    # Compute the x and y coordinates of arc1 to plot it
    x_arc1 = xc1 + R * np.cos(np.linspace(theta1s - tau1 * pi*0.5, theta1e - tau1 * pi*0.5, 100))
    y_arc1 = yc1 + R * np.sin(np.linspace(theta1s - tau1 * pi/2, theta1e - tau1 * pi*0.5, 100))

    # Compute the x and y coordinates of arc3 to plot it
    x_arc3 = xc2 + R * np.cos(np.linspace(theta3s - tau2 * pi*0.5, theta3e - tau2 * pi*0.5, 100))
    y_arc3 = yc2 + R * np.sin(np.linspace(theta3s - tau2 * pi*0.5, theta3e - tau2 * pi*0.5, 100))

    # Build the primitives with the turn on-the-spot
    phi1_star = compute_angular_difference_with_turn_direction(theta0, alpha0 - tau1*pi*0.5, tau1)
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

    # Update the strings
    delta0_string = f"""Delta0: {alpha0:.6f}\nd2: {d2:.6f}\nd3_star: {d3_star:.6f} 
        \nInequality value: {ineq_value:.6f}\nIota4_star: {iota4_star:.6f}
        \nThetat: {thetat:.6f}\n\nTotal time not mirrored: {total_time_3_man_point:.6f}"""
    
    max_index = np.argmax(alpha_vector)
    half_alpha = alpha_vector[max_index:]
    inverted_alpha = np.flip(half_alpha)

    max_index = np.argmax(d2_vector)
    half_d2 = d2_vector[max_index:]
    inverted_d2 = np.flip(d2_vector)

    # Create vectors for forward and angular velocity
    v_vector = np.ones(6) * v_max
    omega_vector = np.zeros(6)
    time_vector = np.zeros(6)
    time_vector[0] = 0
    time_vector[1] = abs(iota1)/omega_max
    time_vector[2] = time_vector[1]
    time_vector[3] = abs(iota1)/omega_max + d2/v_max
    time_vector[4] = time_vector[3]
    time_vector[5] = abs(iota1)/omega_max + d2/v_max + abs(iota3)/omega_max

    omega_vector[0] = tau1*omega_max
    omega_vector[1] = tau1*omega_max
    omega_vector[4] = tau2*omega_max
    omega_vector[5] = tau2*omega_max


    # Compute the optimal solution
    # xs, ys, ts, vs, omegas = ocp_function(x0, y0, theta0, xt, yt, thetat, v_min, v_max, omega_min, omega_max)
    # xs, ys, ts, vs, omegas, xs_ctrl_grid, ys_ctrl_grid, ts_ctrl_grid, vs_ctrl_grid, omegas_ctrl_grid = ms_ocp_function(x0, y0, theta0, xt, yt, thetat, v_min, v_max, omega_min, omega_max)

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
    
    figure1 = {
        'data': [
            go.Scatter(
                x=[x0, x0 + c*cos(alpha0 - tau2*beta)], y=[y0, y0+c*sin(alpha0 - tau2*beta)], mode='lines',
                line=dict(color='black', width=0.7),
                name='Tangent 1'
            ),
        # Add the point at (x0, y0)
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
            shapes = [C1, C2, C1_star],
            annotations = [initial_arrow, final_arrow],
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
        # Add the path from ocp
        # go.Scatter(
        #     x=xs, y=ys, mode='lines',
        #     line=dict(color='red', width=2, dash='dash'),
        #     name='Ocp path'
        #     ), 
        ],
        'layout': go.Layout(
            title=f'Path',
            shapes = [C1, C2, C1_star],
            annotations = [initial_arrow, final_arrow],
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
            x=theta0_vector, y=alpha_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Angle Alpha'
            ),
        # Add the path
        go.Scatter(
            x=theta0_vector, y=inverted_alpha, 
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='Angle Alpha'
            ),
        go.Scatter(
            x=theta0_vector, y=alpha0_vector, 
            mode='lines',
            line=dict(color='red', width=1),
            name='Angle Alpha0'
            ),
        go.Scatter(
            x=[theta0], y=[alpha], mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Alpha'
            )
        ],
        'layout': go.Layout(
            title=f'Angle Alpha',
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
                #Add the path
        # go.Scatter(
        #     x=theta0_vector, y=d2_vector**2 - d3_star_vector**2, 
        #     mode='lines',
        #     line=dict(color='green', width=1),
        #     name='d2 - d3_star'
        # ),
        #         #Add the path
        # go.Scatter(
        #     x=theta0_vector, y=expr_to_check, 
        #     mode='lines',
        #     line=dict(color='red', width=1, dash = 'dash'),
        #     name='d2 - d3_star'
        # ),
        #         go.Scatter(
        #     x=theta0_vector, y=d2_vector - d3_star_vector, 
        #     mode='lines',
        #     line=dict(color='green', width=1),
        #     name='(d2 - d3_star)^2'
        # ),
        # go.Scatter(
        #     x=theta0_vector, y=total_ang_disp_vector, 
        #     mode='lines',
        #     line=dict(color='black', width=1),
        #     name='Epsilon*R'
        # ),
        # go.Scatter(
        #     x=theta0_vector, y=(sum_angles_4_man - sum_angles_3_man)*R, 
        #     mode='lines',
        #     line=dict(color='red', width=1),
        #     name='difference in total angular displacement'
        # ),
        go.Scatter(
            x=theta0_vector, y= d3_star_vector + abs(phi1_star_minus_tau1)*2*R, 
            mode='lines',
            line=dict(color='green', width=1),
            name='2*eps*R'
        ),
        # go.Scatter(
        #     x=theta0_vector, y=sum_angles_4_man, 
        #     mode='lines',
        #     line=dict(color='green', width=1, dash='dash'),
        #     name='Epsilon*R'
        # ),
        # Add the path
        go.Scatter(
            x=theta0_vector, y=d2_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='Distance d2'
        ),
        # go.Scatter(
        #     x=theta0_vector, y= np.sqrt(d3_star**2 - 2*d3_star*np.cos(epsilon_vector) + 2*d3_star - 2*np.cos(epsilon_vector) + 2), #np.sqrt(2*R**2 + d3_star**2 - 2*R*d3_star - 2*R*(R+d3_star) * np.cos(epsilon_vector)), 
        #     mode='lines',
        #     line=dict(color='red', width=2, dash='dash'),
        #     name='Distance d2'
        # ),
        # # Add the path
        # go.Scatter(
        #     x=theta0_vector, y=inverted_d2, 
        #     mode='lines',
        #     line=dict(color='blue', width=2, dash='dash'),
        #     name='Distance d2'
        # ),
        go.Scatter(
            x=theta0_vector, y=d3_star_vector, 
            mode='lines',
            line=dict(color='black', width=1),
            name='Distance d3_star'
        ),
        # go.Scatter(
        #     x=[theta0], y=[d2], mode='markers',
        #     marker=dict(color='green', size=10, symbol='circle'),
        #     name='d2'
        #     )
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
        go.Scatter(
            x=theta0_vector, y=np.zeros(N), 
            mode='lines',
            line=dict(color='red', width=1),
            name='Zero horizontal line'
            )
        ],
        'layout': go.Layout(
            title=f'Inequality value: d2 - d3_star',
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
        go.Scatter(
            x=theta0_vector, y=total_time_4_man, 
            mode='lines',
            line=dict(color='blue', width=1),
            name='Total time with tots'
            ),
        # go.Scatter(
        #     x=theta0_vector, y=total_time_4_man_minus_tau1, 
        #     mode='lines',
        #     line=dict(color='red', width=1, dash='dash'),
        #     name='Total time with tots on opposite direction'
        #     ),
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
            title=f'Total time for 3 and 4 maneuvers',
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
        # go.Scatter(
        #     x=theta0_vector, y=abs(iota2_star) * np.ones(N), 
        #     mode='lines',
        #     line=dict(color='red', width=1),
        #     name='iota2_star'
        #     ),
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

    # Create figure forward velocity
    figure01 = {
        'data': [
        # Add the lower bound
        go.Scatter(
            x=time_vector, y=v_min*np.ones(time_vector.size), 
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            name='v_min'
            ),
        # Add the upper bound
        go.Scatter(
            x=time_vector, y=v_max*np.ones(time_vector.size), 
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            name='v_max'
            ),
        # Add the velocity
        go.Scatter(
            x=time_vector, y=v_vector, 
            mode='lines',
            line=dict(color='green', width=2),
            name='v'
            ),
        # Add the velocity
        # go.Scatter(
        #     x=ts, y=vs, 
        #     mode='lines',
        #     line=dict(color='red', width=2, dash='dash'),
        #     name='v'
        #     )
        ],
        'layout': go.Layout(
            title=f'Forward velocity',
            shapes = [], 
            hovermode='closest',
            showlegend=False,
            autosize=True
            # margin=dict(l=50, r=50, b=50, t=50)  # Optional: adjust the plot's margins
        )
    }

    # Create figure forward velocity
    figure02 = {
        'data': [
        # Add the lower bound
        go.Scatter(
            x=time_vector, y=omega_min*np.ones(time_vector.size), 
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            name='omega_min'
            ),
        # Add the upper bound
        go.Scatter(
            x=time_vector, y=omega_max*np.ones(time_vector.size), 
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            name='omega_max'
            ),
        # Add the angular velocity
        go.Scatter(
            x=time_vector, y=omega_vector, 
            mode='lines',
            line=dict(color='green', width=1),
            name='omega'
            ),
        # # Add the angular velocity
        # go.Scatter(
        #     x=ts, y=omegas, 
        #     mode='lines',
        #     line=dict(color='red', width=1, dash='dash'),
        #     name='omega'
        #     ),
        ],
        'layout': go.Layout(
            title=f'Angular velocity',
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

    return figure1, figure11, figure3, figure2, figure4, figure5, figure6, figure01, figure02, delta0_string




# Run the app
if __name__ == '__main__':
    app.run(debug=True)