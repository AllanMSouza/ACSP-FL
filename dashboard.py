import dash
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from dash_bootstrap_components.themes import FLATLY, BOOTSTRAP # bootstrap theme
import plotly.graph_objs as go

import plotly.express as px
import pandas as pd

global DATASET  
global SOLUTION 

DATASET  = 'UCIHAR'
SOLUTION = 'FedAvg-None'

# app = Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.LUMEN])

select_solution =  html.Div([dbc.Row([
    dbc.Col(dbc.Select(
        id="dataset-select",
        placeholder='Select the dataset',
        required=True,
        options=[{'label' : 'UCI-HAR', 'value' : 'UCIHAR'},
                 {'label' : 'MotionSense', 'value' : 'MotionSense'},
                 {'label' : 'ExtraSensory', 'value' : 'ExtraSensory'}
                 ],
        # "Option 1", "Option B", "Option III", 4],
        # "Option III",
        
    ), md=4),
    dbc.Col(dbc.Select(
        id="solution-select",
        placeholder='Select the solution',
        required=True,
        options=[{'label' : 'FedAvg',     'value' : 'FedAvg-None'},
                 {'label' : 'POC',        'value' : 'POC-POC-0.5'},
                 {'label' : 'DEEV',       'value' : 'DEEV-DEEV-0.01'},
                 {'label' : 'ACSP-FL',    'value' : 'DEEV-PER-DEEV-0.01'},
                 {'label' : 'ACSP-FL LR', 'value' : 'DEEV-PER-SHARED-DEEV-0.01'},
                 ],
        # "Option 1", "Option B", "Option III", 4],
        # "Option III",
        
    ),md=4)
    ]),
    html.Br(),]
)
     

@app.callback(Output("dataset-select", "value"), Input("dataset-select", "value"),)
def select_dataset(select_value):
    global DATASET 
    DATASET = select_value
    return select_value


@app.callback(Output("solution-select", "value"), Input("solution-select", "value"),)
def solution_fl(select_value):
    global SOLUTION 
    SOLUTION = select_value
    return select_value

def create_server_plot():
    # server_df = pd.read_csv('logs/MotionSense/{SOLUTION}/DNN/server.csv',  
    #                     names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
    # fig = px.line(server_df, x='round', y='acc', markers=True, line_color=dict(color="#0000ff"))
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=server_df['round'], y=server_df['acc'], name='Accuracy',
    #                          line=dict(color='#ff7519', width=4), line_shape='spline'))
    fig.add_trace(go.Scatter(x=[], y=[], name='Accuracy',
                             line=dict(color='#ff7519', width=4), line_shape='spline'))

    fig.update_layout(
        # plot_bgcolor='white',
        #paper_bgcolor = 'white',
        margin=dict(t=10,l=10,b=10,r=10),
        xaxis_title='Communication Round',
        yaxis_title='Test Accuracy (%)',
        font=dict(size=15, color='#000000'),

    )
    # fig.update_xaxes(gridcolor='black', griddash='dash', )
    # fig.update_yaxes(gridcolor='black', griddash='dash', )
    return fig

def create_network_plot():
    # client_df = pd.read_csv('logs/MotionSense/{SOLUTION}/DNN/train_client.csv',
    #                     names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
    fig = go.Figure()
    #fig = px.bar(client_df, x='cid', y='param')
    # fig.add_trace(go.Bar(x=client_df['cid'], y=client_df['param'], 
    #                     marker_color='#071f4a'))
    fig.add_trace(go.Bar(x=[], y=[], marker_color='#ff7519'))

    fig.update_layout(
        # plot_bgcolor='white',
        paper_bgcolor = 'white',
        margin=dict(t=10,l=10,b=10,r=10),
        xaxis_title='Client id (#)',
        yaxis_title='Tx Bytes',
        font=dict(size=15, color='#000000'),
    )
    # fig.update_xaxes(gridcolor='black', griddash='dash', )
    # fig.update_yaxes(gridcolor='black', griddash='dash', )
    return fig

def create_latency_plot():
    # client_df = pd.read_csv('logs/MotionSense/{SOLUTION}/DNN/train_client.csv',
    #                     names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
    # df_grouped = client_df.groupby('round').max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[],
                             line=dict(color='#ff7519', width=4), line_shape='spline'))
    fig.update_layout(
        # plot_bgcolor='white',
        paper_bgcolor = 'white',
        margin=dict(t=10,l=10,b=10,r=10),
        xaxis_title='Communication Round',
        yaxis_title='Latency (s)',
        font=dict(size=15, color='#000000')
    )
    # fig.update_xaxes(gridcolor='black', griddash='dash', )
    # fig.update_yaxes(gridcolor='black', griddash='dash', )
    return fig

def create_selection_plot():
    #1, 15, 93424, 1.7241395711898804, 0.29165393114089966
    # client_df = pd.read_csv('logs/MotionSense/{SOLUTION}/DNN/evaluate_client.csv',
    #                     names=['round', 'cid', 'size', 'loss', 'acc'])
    fig = go.Figure()
    #fig = client_df, x='cid', y='selected'
    # client_df = client_df[client_df['round'] == 100]
    fig.add_trace(go.Bar(x=[], y=[], 
                         meta='Rainbow'))
    fig.update_layout(
        # plot_bgcolor='white',
        paper_bgcolor = 'white',
        margin=dict(t=10,l=10,b=10,r=10),
        xaxis_title='Client id (#)',
        yaxis_title='Test Accuracy (%)',
        font=dict(size=15, color='#000000'),
    )
    # fig.update_xaxes(gridcolor='black', griddash='dash', )
    # fig.update_yaxes(gridcolor='black', griddash='dash', )
    return fig

tabs = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Summary", tab_id="tab-1"),
                    dbc.Tab(label="Server", tab_id="tab-2"),
                    dbc.Tab(label="Clients", tab_id="tab-3"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),
    ]
)


@app.callback(
    Output("card-content", "children"), [Input("card-tabs", "active_tab")]
)
def tab_content(active_tab):
    if active_tab == "tab-1":
        return grid

server_plot = dbc.Card([
                dbc.CardHeader(children=['Aggregated Teste Accuracy '],
                               style={'background-color' : '#071f4a', 'font-size': '24px', 'color': '#FFFFFF'}),

                dbc.CardBody([
                     dcc.Interval(id="interval"),
                     dcc.Graph(figure=create_server_plot(), id='server_plot', animate=True),
                ]),
                
              ], class_name='mb-3',)  

communication_plot    = dbc.Card([
                        dbc.CardHeader(children=['Network Usage'],
                                       style={'background-color' : '#071f4a', 'font-size': '24px', 'color': '#FFFFFF'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_network_plot(), 
                                id='communication_plot',
                                animate=True
                            ),
                        ])
                     ])

latency_plot         = dbc.Card([
                        dbc.CardHeader(children=['Latency'],
                                       style={'background-color' : '#071f4a', 'font-size': '24px', 'color': '#FFFFFF'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_latency_plot(), 
                                id='latency_plot',
                                animate=True
                            ),
                        ])
                     ])

client_selection_plot = dbc.Card([
                        dbc.CardHeader(children=['Client Selection'],
                                       style={'background-color' : '#071f4a', 'font-size': '24px', 'color': '#FFFFFF'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_selection_plot(), 
                                id='client_plot',
                                animate=True
                            ),

                        ])
                     ])

HIAAC_LOGO = "https://hiaac.unicamp.br/wp-content/themes/hiaac_portal/assets/images/logo.svg"
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=HIAAC_LOGO, height="60px")),
                        dbc.Col(dbc.NavbarBrand("FL-H.IAAC", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://plotly.com",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        ], fluid=True
    ),
    color="light",
    dark=False,
)

grid = html.Div([
      select_solution,
      html.Div([
          server_plot
          ]),
      #linha
        dbc.Row(
            [
              #coluna01
              dbc.Col(communication_plot, md=4),                  
              #coluna02
              dbc.Col(latency_plot, md=4),
              #coluna03
              dbc.Col(client_selection_plot, md=4),
            ]
        ),
    ]
)

app.layout = html.Div(
    [
        dbc.Container(
            [  
                #navbar,
                tabs,
            ], fluid=True
        )
    ]
)


@app.callback(
    [Output("server_plot", "figure")],
    [Input("interval", "n_intervals")]
)
def update_server_plot(n_intervals):
    global DATASET
    global SOLUTION
    server_df = pd.read_csv(f'logs/{DATASET}/{SOLUTION}/DNN/server.csv',  
                         names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
    server_df = server_df[server_df['round'] <= n_intervals]
    fig = go.Figure()
    trace = go.Scatter(x=server_df['round'], y=server_df['acc'], name='Accuracy',
                              line=dict(color='#ff7519', width=4), line_shape='spline')


    return [go.Figure(data=trace, layout=go.Layout(
            xaxis=dict(range=[0, max(server_df['round'])]),
            yaxis = dict(range=[min(server_df['acc']), max(server_df['acc'])+0.1]),))]


@app.callback(
    [Output("communication_plot", "figure")],
    [Input("interval", "n_intervals")]
)
def update_network_plot(n_intervals):
    global DATASET
    global SOLUTION
    client_df = pd.read_csv(f'logs/{DATASET}/{SOLUTION}/DNN/train_client.csv',
                        names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
    client_df2 = client_df[client_df['round'] <= n_intervals].copy()
    trace = go.Bar(x=client_df2['cid'], y=client_df2['param'], marker_color='#ff7519')

    client_df2.reset_index()
    max_value = client_df2.groupby('round').max()

    return [go.Figure(data=trace, layout=go.Layout(
            xaxis=dict(range=[- 1, max(client_df2['cid']) + 1]),
            yaxis = dict(range=[min(client_df2['param']), max(max_value['param'].cumsum())+50]),
            ))
            ]

@app.callback(
    [Output("latency_plot", "figure")],
    [Input("interval", "n_intervals")]
)
def update_latency_plot(n_intervals):
    global DATASET
    global SOLUTION
    server_df = pd.read_csv(f'logs/{DATASET}/{SOLUTION}/DNN/server.csv',  
                         names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
    
    server_df = server_df[server_df['round'] <= n_intervals]

    server_df['latency'] = server_df['timestamp'].diff()
    server_df.fillna(0, inplace=True)

    trace = go.Scatter(x=server_df['round'], y=server_df['latency'],
                              line=dict(color='#ff7519', width=4), line_shape='spline')

    # print(server_df['latency'])
    # time = server_df['timestamp'].values[-1] - server_df['timestamp'].values[-2]

    return [go.Figure(data=trace, layout=go.Layout(
            xaxis=dict(range=[0, max(server_df['round'])]),
            yaxis = dict(range=[0, max(server_df['latency'])+2]),))]

@app.callback(
    [Output("client_plot", "figure")],
    [Input("interval", "n_intervals")]
)
def update_client_plot(n_intervals):
    global DATASET
    global SOLUTION
    client_df = pd.read_csv(f'logs/{DATASET}/{SOLUTION}/DNN/evaluate_client.csv',
                        names=['round', 'cid', 'size', 'loss', 'acc'])
    
    
    client_df = client_df[client_df['round'] == n_intervals]
    client_df.sort_values(by=['cid'], inplace=True)
    trace     = go.Bar(x=client_df['cid'], y=client_df['acc'], 
                         marker_color='#071f4a')

    return [go.Figure(data=trace, layout=go.Layout(
            xaxis=dict(range=[-1 , max(client_df['cid']) + 1]),
            yaxis = dict(range=[0, 1]),
            ))
            ]



if __name__ == '__main__':
    app.run(debug=True)