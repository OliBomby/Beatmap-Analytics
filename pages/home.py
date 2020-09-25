import numpy as np
import pandas as pd
import slider
import base64
import io
import math

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from app import app


layout = html.Div([
     html.H1("Beatmap Analytics", className="display-3", style={'textAlign': 'center'}),
     html.P(
         "Visualizing beatmaps is just a *click* away!",
         className="lead",
         style={'textAlign': 'center', 'margin': '10px'}
     ),
     html.Hr(className="my-2"),
     html.P(
         "Beatmap Analytics is an app for getting analytics about space usage in beatmaps.",
         style={'textAlign': 'center', 'margin': '30px 30px 0px'}
     ),
     html.P(
         "Upload one or more .osu files files.",
         style={'textAlign': 'center', 'margin': '0px 30px 30px'}
     ),
     dcc.Upload(
         id='upload-data',
         children=html.Div([
             'Drag and Drop or ',
             html.A('Select File')
         ]),
         style={
             'width': '600px',
             'height': '60px',
             'lineHeight': '60px',
             'borderWidth': '1px',
             'borderStyle': 'dashed',
             'borderRadius': '5px',
             'textAlign': 'center',
             'margin': 'auto'
         },
         multiple=True
     ),
     html.Hr(className="my-2"),

     html.Div(id='output-data-upload'),
     html.Div(id='output-heatmap'),
])


def get_points_from_beatmap(beatmap, easy=False, hard_rock=False, double_time=False):
    def generator():
        beatmap.slider_tick_rate = 4
        radius = slider.beatmap.circle_radius(beatmap.cs(easy=easy, hard_rock=hard_rock))
        slider_radius = radius * 1.5
        time_multiplier = 2.0 / 3.0 if double_time else 4.0 / 3.0 if easy else 1.0
        for h in beatmap.hit_objects(spinners=False):
            time = time_multiplier * h.time.total_seconds()
            pos_x = h.position.x
            pos_y = h.position.y

            yield [time, pos_x, pos_y, radius, False]

            if isinstance(h, slider.beatmap.Slider):
                for point in h.tick_points:
                    time = time_multiplier * point.offset.total_seconds()
                    yield [time, point.x, point.y, slider_radius, True]

    result = np.array([row for row in generator()])
    return result


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    if '.osu' in filename:
        beatmap = slider.Beatmap.from_file(io.StringIO(decoded.decode("utf-8-sig")))

        return get_points_from_beatmap(beatmap)
    else:
        return html.Div(["You have to upload in .osu format."]), dash.no_update


@app.callback([Output('output-data-upload', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        point_arrays = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]

        points = np.concatenate(point_arrays, axis=0)
        df_points = pd.DataFrame(data=points[:, 1:], index=points[:, 0], columns=['x', 'y', 'radius', 'is_tick'])

        graph = dcc.Graph(figure=make_scatter(df_points))

        left, right = get_lr_balance(df_points)
        obj_in, obj_out = get_io_balance(df_points)

        return [html.Div(children=[
            html.P(
                "Object balance: Inner: %s%%, Outer: %s%%" % (obj_in * 100, obj_out * 100),
                style={'textAlign': 'center', 'margin': '30px 30px 0px'}
            ),
            html.P(
                "Object balance: Left: %s%%, Right: %s%%" % (left * 100, right * 100),
                style={'textAlign': 'center'}
            ),
            graph,
        ])]

    return dash.no_update


@app.callback([Output('output-heatmap', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        point_arrays = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]

        points = np.concatenate(point_arrays, axis=0)
        df_points = pd.DataFrame(data=points[:, 1:], index=points[:, 0], columns=['x', 'y', 'radius', 'is_tick'])

        graph2 = dcc.Graph(figure=make_heatmap(df_points))

        return [html.Div(children=[
            graph2,
        ])]

    return dash.no_update


def get_lr_balance(df_points):
    left_objects = len(df_points[df_points['x'] < 256].index)
    right_objects = len(df_points[df_points['x'] > 256].index)

    total = left_objects + right_objects
    return left_objects / total, right_objects / total


def get_io_balance(df_points):
    # Bounds for a proportional rectangle in the centre of the playfield with exactly half the area
    left = 74.95
    right = 437.05
    top = 327.75
    bottom = 56.25

    inner_objects = len(df_points[((df_points['x'] < right) & (df_points['x'] > left)) & (
                (df_points['y'] < top) & (df_points['y'] > bottom))].index)
    outer_objects = len(df_points[((df_points['x'] > right) | (df_points['x'] < left)) | (
                (df_points['y'] > top) | (df_points['y'] < bottom))].index)

    total = inner_objects + outer_objects
    return inner_objects / total, outer_objects / total


def make_scatter(points):
    fig = px.scatter(points, x="x", y="y", size="radius", opacity=0.1, range_x=(0, 512), range_y=(0, 384))

    fig.update_layout(
        title={
            'text': "Scatter plot",
            'y': 1.0,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis=dict(
            range=[0, 512],
        ),
        yaxis=dict(
            range=[0, 384],
            autorange="reversed"
        ),
        width=2 * 512,
        height=2 * 384,
    )

    return fig


def make_heatmap(points):
    img_width = 512
    img_height = 384
    visual_span = 30

    # algorithm from Blignaut, Pieter. (2010). Visual span and other parameters for the generation of heatmaps.
    # Eye Tracking Research and Applications Symposium (ETRA). 125-128. 10.1145/1743666.1743697.
    weights = np.zeros((img_height, img_width))
    for index, fixation in points.iterrows():
        x = fixation['x']
        y = fixation['y']
        weight = 100
        for col in range(int(x) - visual_span, int(x) + visual_span):
            if 0 <= col < img_width:
                for row in range(int(y) - visual_span, int(y) + visual_span):
                    if 0 <= row < img_height:
                        distance = ((col - x) ** 2 + (row - y) ** 2) ** 0.5
                        if distance <= visual_span:
                            probability = math.exp(-(distance ** 2) / (2 * (0.17 * visual_span) ** 2))
                            weights[row][col] += weight * probability

    fig_image = go.Figure(
        data=go.Heatmap(
            opacity=1,
            z=np.flip(weights, 0)
        )
    )
    fig_image.update_layout(
        title={
            'text': "Heatmap",
            'y': 1.0,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis=dict(
            range=[0, 512],
        ),
        yaxis=dict(
            range=[0, 384],
            autorange="reversed"
        ),
        width=2 * 512,
        height=2 * 384,
    )

    return fig_image
