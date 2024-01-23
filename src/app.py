import dash
import pandas as pd
from dash import Dash, dash_table, dcc, html, Input, Output, State
import plotly.express as px
import os
from datetime import timedelta, datetime as dt

app = Dash(__name__)
server = app.server

# LOAD DATA
# df = px.data.gapminder()
df = pd.read_csv('../data/final_dashboard_df.csv')
df.day = pd.to_datetime(df.day)
df.datetime_hourly = pd.to_datetime(df.datetime_hourly)

# FORMAT DATES FOR SLIDER
date_data = {'Date': pd.date_range(start='2021-01-01', end = '2021-12-31'),
             'Value': [i for i in range(365)]}
date_df = pd.DataFrame(date_data)

# Initialize the Dash app
# app = dash.Dash(__name__)

# App layout
# Create marks for the first day of each month
marks = {i: date.strftime('%Y-%m-%d') for i, date in enumerate(date_df['Date']) if date.day == 1}

# App layout
app.layout = html.Div([
    html.Div([
        dcc.RangeSlider(
            id='date-slider',
            marks=marks,
            min=0,
            max=len(date_df['Date']) - 1,
            value=[0, len(date_df['Date']) - 1],
            step=1,
            allowCross=False
        ),
        dcc.Graph(id='line-chart')
    ]),
#     html.Div([
#     dcc.Graph(id='ldc-chart'),
#     dcc.Graph(id='heat-map')
# ])
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

    fig = px.line(filtered_df, x='datetime_hourly', y='energy', color='Location', line_group='Location',title='Line Chart with Daily Date Slider')
    return fig

# @app.callback(
#     Output('line-chart', 'figure'),
#     [Input('date-slider', 'value')]
# )
# def update_chart(selected_dates):
#     start_date = date_df['Date'].iloc[selected_dates[0]]
#     end_date = date_df['Date'].iloc[selected_dates[1]]
#
#     filtered_df = df[(df['day'] >= start_date) & (df['day'] <= end_date)]
#
#     fig = px.line(filtered_df, x='datetime_hourly', y='energy', color='Location', line_group='Location',title='Line Chart with Daily Date Slider')
#     return fig

if __name__ == "__main__":
    app.run_server(debug=True)

# range_slider = dcc.RangeSlider(
#     value=[1987, 2007],
#     step=5,
#     marks={i: str(i) for i in range(1952, 2012, 5)},
# )
#
# dtable = dash_table.DataTable(
#     columns=[{"name": i, "id": i} for i in sorted(df.columns)],
#     sort_action="native",
#     page_size=10,
#     style_table={"overflowX": "auto"},
# )
#
# download_button = html.Button("Download Filtered CSV", style={"marginTop": 20})
# download_component = dcc.Download()
#
# app.layout = html.Div(
#     [
#         html.H2("Gapminder Data Download", style={"marginBottom": 20}),
#         download_component,
#         range_slider,
#         download_button,
#         dtable,
#     ]
# )
#
#
# @app.callback(
#     Output(dtable, "data"),
#     Input(range_slider, "value"),
# )
# def update_table(slider_value):
#     if not slider_value:
#         return dash.no_update
#     dff = df[df.year.between(slider_value[0], slider_value[1])]
#     return dff.to_dict("records")
#
#
# @app.callback(
#     Output(download_component, "data"),
#     Input(download_button, "n_clicks"),
#     State(dtable, "derived_virtual_data"),
#     prevent_initial_call=True,
# )
# def download_data(n_clicks, data):
#     dff = pd.DataFrame(data)
#     return dcc.send_data_frame(dff.to_csv, "filtered_csv.csv")
#

# if __name__ == "__main__":
#     app.run_server(debug=True)