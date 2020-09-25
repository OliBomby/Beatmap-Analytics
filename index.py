import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from pages import home

page_mapping = {
    '/': home.layout
}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=home.layout, style={'margin-top': '56px'})
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])
def display_page(pathname):
    if pathname in page_mapping:
        return page_mapping[pathname]
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)
