import dash
import pandas as pd
import numpy as np
from dash import Dash, dash_table, dcc, html, Input, Output, State
import plotly.express as px

app = Dash(__name__)
server = app.server

# LOAD DATA
df = pd.read_csv('../data/final_dashboard_df.csv')
df.day = pd.to_datetime(df.day)
df.datetime_hourly = pd.to_datetime(df.datetime_hourly)

# fill na w/ mean
# numeric_cols = df.select_dtypes(include='number').columns
# df[numeric_cols] = df.groupby('Location')[numeric_cols].transform(lambda col: col.fillna(col.mean()))

# custom color scale
color_scale_dash = [
    (0.0, '#FFB100'), # yellow
    (1.0, '#202850') # blue
]

# FORMAT DATES FOR SLIDER
date_data = {'Date': pd.date_range(start='2021-01-01', end='2021-12-31'),
             'Value': [i for i in range(365)]}
date_df = pd.DataFrame(date_data)


# App layout
# Create marks for the first day of each month
marks = {i: date.strftime('%Y-%m-%d') for i, date in enumerate(date_df['Date']) if date.day == 1}

# App layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='line-chart'),
        dcc.RangeSlider(
            id='date-slider',
            marks=marks,
            min=0,
            max=len(date_df['Date']) - 1,
            value=[0, len(date_df['Date']) - 1],
            step=1,
            allowCross=False
        ),
    ]),
    html.Div([
        dcc.Graph(id='demand-chart'),
        dcc.Graph(id='ldc-chart'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='vt-heat-map'),
        dcc.Graph(id='nema-heat-map')
    ], style={'display': 'inline-block', 'width': '49%'})
])


# Callback to update the chart based on the date slider
@app.callback(
    Output('line-chart', 'figure'),
    [Input('date-slider', 'value')]
)
def update_chart(selected_dates):
    start_date = date_df['Date'].iloc[selected_dates[0]]
    end_date = date_df['Date'].iloc[selected_dates[1]]

    filtered_df = df[(df['day'] >= start_date) & (df['day'] <= end_date)]

    fig = px.line(filtered_df, x='datetime_hourly', y='energy', color='Location',
                  color_discrete_map={'NEMA': color_scale_dash[1][1], 'VT': color_scale_dash[0][1]},
                  line_group='Location',
                  title='A Quick Energy Dashboard').update_layout(
        xaxis_title = '',
        yaxis_title = 'Load (MWh)'
    )

    fig.update_layout(height=400)

    return fig


@app.callback(
Output('demand-chart', 'figure'),
    Output('ldc-chart', 'figure'),
    Output('vt-heat-map', 'figure'),
    Output('nema-heat-map', 'figure'),
    [Input('date-slider', 'value')]
)
def update_chart(selected_dates):
    start_date = date_df['Date'].iloc[selected_dates[0]]
    end_date = date_df['Date'].iloc[selected_dates[1]]
    filtered_df = df[(df['day'] >= start_date) & (df['day'] <= end_date)]

    # Re-Size Heatmap Zone
    vt_sum_lmp_cost = sum(filtered_df[(filtered_df.Location == 'VT') & (filtered_df.total_lmp.notnull())].total_lmp)
    nema_sum_lmp_cost = sum(filtered_df[(filtered_df.Location == 'NEMA') & (filtered_df.total_lmp.notnull())].total_lmp)
    filtered_df['energy_cost_pct'] = filtered_df.apply(
        lambda x: x['total_lmp'] / vt_sum_lmp_cost if x['Location'] == 'VT' else x['total_lmp'] / nema_sum_lmp_cost,
        axis=1)

    # df for heat-map
    hm1 = filtered_df[filtered_df.Location == 'NEMA'][['day', 'hour', 't_f']]
    hm1 = hm1.pivot_table(index='hour',columns='day', values='t_f', aggfunc='mean')

    hm2 = filtered_df[filtered_df.Location == 'NEMA'][['day','hour','lmp']]
    hm2 = hm2.pivot_table(index='hour',columns='day',values='lmp', aggfunc='mean')

    # df for ldc
    vt_ldc = filtered_df[(filtered_df.Location == 'VT') & (filtered_df.energy.notnull())].sort_values('energy', ascending=False).energy.reset_index(drop=True)
    nema_ldc = filtered_df[(filtered_df.Location == 'NEMA') & (filtered_df.energy.notnull())].sort_values('energy', ascending=False).energy.reset_index(drop=True)

    vt_ldc = vt_ldc / max(vt_ldc)
    nema_ldc = nema_ldc/max(nema_ldc)

    ldc_index = np.linspace(0,1,len(vt_ldc))
    ldc = pd.DataFrame(list(zip(ldc_index,vt_ldc,nema_ldc)),columns=['ldc_index', 'VT', 'NEMA'])

    # PLOTS
    fig_dem = px.scatter(filtered_df, x='t_f', y='energy_normalized',color='Location',
                      color_discrete_map={'NEMA': color_scale_dash[1][1], 'VT': color_scale_dash[0][1]},
                      title='Demand vs. Temperature').update_layout(
        xaxis_title = 'temperature (F)',
        yaxis_title = 'normalized load'
    )
    fig_ldc = px.line(ldc,x='ldc_index', y=['VT','NEMA'],
                      color_discrete_map={'NEMA': color_scale_dash[1][1],
                                          'VT': color_scale_dash[0][1]}, title='Load Duration Curve').update_layout(
        xaxis_title = 'index',
        yaxis_title = 'percent of peak'
    )
    fig_vt = px.imshow(hm1, x=hm1.columns, y=hm1.index,
                       title='Temperature Heat Map').update_layout(
        xaxis_title = 'day',
        yaxis_title = 'hour')
    fig_nema = px.imshow(hm2, x=hm2.columns, y=hm2.index, title='LMP Heat Map').update_layout(
        xaxis_title = 'day',
        yaxis_title = 'hour')

    # sizing
    fig_dem.update_layout(height=350)
    fig_ldc.update_layout(height=350)
    fig_vt.update_layout(height=350)
    fig_nema.update_layout(height=350)

    return fig_dem, fig_ldc, fig_vt, fig_nema


if __name__ == "__main__":
    app.run_server(debug=True, port=8071)
    # app.run_server(debug=True)
