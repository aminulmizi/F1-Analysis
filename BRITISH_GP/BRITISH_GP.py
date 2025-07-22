from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from scipy.stats import linregress

# Load data
df = pd.read_csv('BRITISHGP_LAPTIMES.csv')
df['Lap'] = df['Lap'].astype(int)

# Drivers & stints definitions
drivers = {'Ollie Bearman': 'white', 'Esteban Ocon': 'red'}
stints = {
    'Full Race': {'slices': {'Ollie Bearman': slice(None), 'Esteban Ocon': slice(None)}},
    'Stint 1': {'slices': {'Ollie Bearman': slice(0, 9), 'Esteban Ocon': slice(0, 16)}},
    'Stint 2': {'slices': {'Ollie Bearman': slice(10, 40), 'Esteban Ocon': slice(17, 41)}},
    'Stint 3': {'slices': {'Ollie Bearman': slice(41, 51), 'Esteban Ocon': slice(42, 51)}}
}

# Race events annotations
race_events = [
    {'lap': 2, 'label': 'VSC Start'}, {'lap': 4, 'label': 'VSC End'},
    {'lap': 6, 'label': 'VSC Start'}, {'lap': 7, 'label': 'VSC End'},
    {'lap': 11, 'label': 'Rain Arrival'}, {'lap': 14, 'label': 'SC Start'},
    {'lap': 16, 'label': 'Rain End'}, {'lap': 17, 'label': 'SC End'},
    {'lap': 18, 'label': 'SC Start'}, {'lap': 21, 'label': 'SC End'}
]

# Helper to get stint data
def get_stint_df(driver, stint_key):
    col = 'Laptime (s) OB' if driver == 'Ollie Bearman' else 'Laptime (s) EO'
    slc = stints[stint_key]['slices'][driver]
    sub = df.iloc[slc][['Lap', col]].dropna().rename(columns={col: 'Time'}).reset_index(drop=True)
    sub['RelLap'] = sub.index + 1
    sub['AbsLap'] = df.iloc[slc]['Lap'].values
    return sub

# Monte Carlo functions
def simulate_race(n_laps, p_rain=0.02, p_sc=0.03, p_rf=0.01,
                  base_mu=90, base_sigma=2,
                  rain_mu=5, rain_sigma=2,
                  sc_time=90,
                  rf_maxlap=10,
                  rf_duration=300,
                  p_rain_exit=0.1,
                  p_sc_exit=0.3,
                  sc_min_laps=1):
    total = 0.0
    in_rain = False
    in_sc = False
    redflag_handled = False
    lap = 1
    sc_lap_counter = 0

    while lap <= n_laps:
        if not redflag_handled and lap <= rf_maxlap and np.random.rand() < p_rf:
            total += rf_duration
            redflag_handled = True

            in_rain = False
            in_sc = False

            lap += 1
            continue

        if not in_sc and np.random.rand() < p_sc:
            in_sc = True
            sc_lap_counter = 0

        if not in_rain and np.random.rand() < p_rain:
            in_rain = True

        if in_sc:
            lt = sc_time
            sc_lap_counter += 1
            if sc_lap_counter >= sc_min_laps and np.random.rand() < p_sc_exit:
                in_sc = False
        else:
            if in_rain:
                mu = base_mu + rain_mu
                sigma = np.sqrt(base_sigma**2 + rain_sigma**2)
                lt = np.random.normal(mu, sigma)
                if np.random.rand() < p_rain_exit:
                    in_rain = False
            else:
                lt = np.random.normal(base_mu, base_sigma)

        total += lt
        lap += 1

    return total


def run_simulation(n_sims, **kwargs):
    return np.array([simulate_race(**kwargs) for _ in range(n_sims)])

# Build app
def serve_app():
    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    
    tabs = dcc.Tabs([
        dcc.Tab(label='Boxplots', children=[
            dbc.Row(dbc.Col(dcc.Dropdown(id='box-stint', options=[{'label': k, 'value': k} for k in stints], value='Full Race'), width=4), className='my-4 justify-content-center'),
            dcc.Graph(id='box-graph', className='p-3'),
            dcc.Graph(id='lap-graph', className='p-3'),
            dcc.Graph(id='delta-graph', className='p-3')
        ]),
        dcc.Tab(label='Stint Analysis', children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='analysis-stint', options=[{'label': k, 'value': k} for k in stints], value='Stint 2'), width=3),
                dbc.Col(dcc.RadioItems(id='axis-type', options=[{'label':'RelLap','value':'RelLap'},{'label':'AbsLap','value':'AbsLap'}], value='RelLap', inline=True), width=3),
                dbc.Col(html.Div([html.Label('Shift Comparison'), dcc.Slider(id='lap-shift', min=-10, max=10, step=1, value=0, tooltip={'always_visible':True})]), width=6)
            ], className='my-4 gx-4 align-items-center'),
            dcc.Graph(id='rollavg-abs-graph', className='p-3'),
            dcc.Graph(id='overlay-compare-graph', className='p-3'),
            html.Ul(id='metric-list')
        ]),
        dcc.Tab(label='What-If Simulation', children=[
            dbc.Row([
                 dbc.Col(html.Div([html.Label('Simulation Runs'), dcc.Input(id='sim-runs', type='number', value=1000, min=100, step=100)]), width=2),
                dbc.Col(html.Div([html.Label('Rain Probability'), dcc.Slider(id='prob-rain', min=0, max=0.2, step=0.01, value=0.02, tooltip={'always_visible':True})]), width=4),
                dbc.Col(html.Div([html.Label('Safety Car Probability'), dcc.Slider(id='prob-sc', min=0, max=0.1, step=0.01, value=0.03, tooltip={'always_visible':True})]), width=4),
                dbc.Col(html.Div([html.Label('Red Flag Probability'), dcc.Slider(id='prob-rf', min=0, max=0.05, step=0.005, value=0.01, tooltip={'always_visible':True})]), width=4)
            ], className='my-3'),
            dbc.CardGroup([
                dbc.Card([dbc.CardHeader('Red Flag Can Occur Up To Lap…'), dbc.CardBody([dbc.Input(id='rf-maxlap', type='number', value=10), dbc.FormText('Lap threshold', color='secondary')])]),
                dbc.Card([dbc.CardHeader('SC Time'), dbc.CardBody([dbc.Input(id='sc-time', type='number', value=90), dbc.FormText('Lap Time Under Safety Car [estimation]', color='secondary')])]),
                dbc.Card([dbc.CardHeader('Rain μ'), dbc.CardBody([dbc.Input(id='rain-pen-mu', type='number', value=5), dbc.FormText('Mean Lap Time Penalty Due to Rain [Added to dry lap time]', color='secondary')])]),
                dbc.Card([dbc.CardHeader('Rain σ'), dbc.CardBody([dbc.Input(id='rain-pen-sig', type='number', value=2), dbc.FormText('Variability of Lap Time Penalty due to Rain', color='secondary')])])
            ], className='mb-4'),
            dcc.Graph(id='sim-graph', className='p-3')
        ])
    ])

    app.layout = dbc.Container([html.H1('F1 Race Analysis'), tabs], fluid=True)

    # Callbacks
    @app.callback(Output('box-graph', 'figure'), Input('box-stint', 'value'))
    def update_box(stint_key):
        traces = []
        for drv, c in drivers.items():
            df_st = get_stint_df(drv, stint_key)
            traces.append(go.Box(x=df_st['Time'], y=[drv]*len(df_st), orientation='h', marker=dict(color=c), boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig = go.Figure(traces)
        fig.update_layout(title=f'{stint_key} Boxplot', template='plotly_dark')
        return fig

    @app.callback(Output('lap-graph', 'figure'), Input('box-stint', 'value'))
    def update_laptimes(_):
        fig = go.Figure()
        for drv in drivers:
            col = 'Laptime (s) OB' if drv == 'Ollie Bearman' else 'Laptime (s) EO'
            fig.add_trace(go.Scatter(x=df['Lap'], y=df[col], mode='lines', name=drv, line=dict(color=drivers[drv], width=2)))
        fig.update_layout(title='Lap Times', template='plotly_dark')
        return fig

    @app.callback(Output('delta-graph', 'figure'), Input('box-stint', 'value'))
    def update_delta(_):
        fig = go.Figure()
        for ev in race_events:
            fig.add_vline(x=ev['lap'], line_dash='dash', opacity=0.7)
            fig.add_annotation(
                x=ev['lap'], y=1.05,
                xref='x', yref='paper',
                text=ev['label'],
                showarrow=False,
                textangle=90,
                font=dict(size=10)
            )
        fig.add_trace(go.Scatter(
            x=df['Lap'],
            y=df['Delta (s)'],
            mode='lines',
            name='Delta',
            line=dict(color='lightblue', width=2)
        ))
        fig.update_layout(
            title='Time Delta',
            template='plotly_dark',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    @app.callback(
        Output('rollavg-abs-graph', 'figure'),
        Output('overlay-compare-graph', 'figure'),
        Output('metric-list', 'children'),
        Input('analysis-stint', 'value'),
        Input('axis-type', 'value'),
        Input('lap-shift', 'value')
    )
    def update_analysis(stint_key, axis, shift):
        fig_abs = go.Figure()
        fig_overlay = go.Figure()
        metrics = []
        for drv in drivers:
            df_st = get_stint_df(drv, stint_key)
            df_st['RollAvg'] = df_st['Time'].rolling(3, min_periods=1).mean()
            fig_abs.add_trace(go.Scatter(x=df_st['AbsLap'], y=df_st['RollAvg'], mode='lines', name=drv, line=dict(width=4)))
            x = df_st[axis] + (shift if drv == 'Esteban Ocon' else 0)
            fig_overlay.add_trace(go.Scatter(x=x, y=df_st['RollAvg'], mode='lines', name=drv, line=dict(width=4)))
            slope = linregress(df_st['RelLap'], df_st['Time']).slope
            metrics.append(html.Li(f"{drv}: Mean {df_st['Time'].mean():.3f}s, Fastest {df_st['Time'].min():.3f}s, Degr {slope:.4f}s/lap"))
        for ev in race_events:
            fig_abs.add_vline(x=ev['lap'], line_dash='dash', opacity=0.7)
            fig_abs.add_annotation(x=ev['lap'], y=0.95, xref='x', yref='paper', text=ev['label'], showarrow=False, textangle=90, font=dict(size=10))
        fig_abs.update_layout(title='Rolling Avg (Absolute Lap)', template='plotly_dark', margin=dict(l=50, r=50, t=50, b=50), height=600)
        fig_overlay.update_layout(title=f'Overlay Rolling Avg (shift={shift})', template='plotly_dark', margin=dict(l=50, r=50, t=50, b=50), height=600)
        return fig_abs, fig_overlay, metrics

    @app.callback(
        Output('sim-graph', 'figure'),
        Input('sim-runs', 'value'),
        Input('prob-rain', 'value'),
        Input('prob-sc', 'value'),
        Input('prob-rf', 'value'),
        Input('rf-maxlap', 'value'),
        Input('sc-time', 'value'),
        Input('rain-pen-mu', 'value'),
        Input('rain-pen-sig', 'value')
    )
    def update_sim(n_sims, p_rain, p_sc, p_rf, rf_max, sc_time, rain_mu, rain_sigma):
        comb = np.concatenate([df['Laptime (s) OB'].dropna(), df['Laptime (s) EO'].dropna()])
        base_mu, base_sig = comb.mean(), comb.std()
        sims = run_simulation(n_sims, n_laps=df['Lap'].max(), p_rain=p_rain, p_sc=p_sc, p_rf=p_rf, base_mu=base_mu, base_sigma=base_sig, rain_mu=rain_mu, rain_sigma=rain_sigma, sc_time=sc_time, rf_maxlap=rf_max)
        counts, bins = np.histogram(sims, bins=50)
        hist = go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=counts)
        pcts = np.percentile(sims, [10, 50, 90])
        lines = [go.Scatter(x=[p, p], y=[0, counts.max()], mode='lines', line=dict(dash='dash'), name=f"{lbl} pct") for p, lbl in zip(pcts, ['10th', '50th', '90th'])]
        fig = go.Figure([hist] + lines)
        fig.update_layout(title='MC Sim', template='plotly_dark')
        return fig

    return app

if __name__ == '__main__':
    serve_app().run(debug=True)
