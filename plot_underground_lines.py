import plotly.graph_objects as go
import plotly.offline as py
from build_data import build_data


def plot_underground_lines(output, stations, underground_lines):
    """
    :param output: Path to output HTML
    :param stations: A mapping between station names and station objects of the name
    :param underground_lines: A mapping between underground lines name and a dictionary containing relevant
                             information about underground lines
    :return: None
    """

    mapbox_access_token = (
        'pk.eyJ1IjoibHVrYXNtYXJ0aW5lbGxpIiwiYSI6ImNpem85dmhwazAy'
        'ajIyd284dGxhN2VxYnYifQ.HQCmyhEXZUTz3S98FMrVAQ'
    )  # 此处的写法只是为了排版，结果为连接在一起的字符串
    layout = go.Layout(
        autosize=True,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=51.5074,  # 伦敦市纬度
                lon=-0.1278  # 伦敦市经度
            ),
            pitch=0,
            zoom=10
        ),
    )
    data = []
    colors = ('blue', 'green', 'yellow', 'purple', 'orange', 'red', 'violet',
              'navy', 'crimson', 'cyan', 'magenta', 'maroon', 'peru')
    for underground_line, color in zip(underground_lines.values(), colors):
        data.extend([
            # 地铁路线
            go.Scattermapbox(
                lat=underground_line['lat'],
                lon=underground_line['lon'],
                mode='lines',
                # 设置路线的参数
                line=go.scattermapbox.Line(
                    width=2,
                    color=color
                ),
                name=underground_line['name'],  # 线路名称，显示在图例（legend）上
                legendgroup=underground_line['name'],
            ),
            go.Scattermapbox(
                lat=[stations[station_name].position[0] for station_name in underground_line['stations']],  # 路线点经度
                lon=[stations[station_name].position[1] for station_name in underground_line['stations']],  # 路线点纬度
                mode='markers',
                text=[stations[station_name].name for station_name in underground_line['stations']],
                # 设置标记点的参数
                marker=go.scattermapbox.Marker(
                    size=6,
                    color=color
                ),
                name=underground_line['name'],
                legendgroup=underground_line['name'],  # 设置与路线同组，当隐藏该路线时隐藏标记点
                showlegend=False  # 不显示图例（legend)
            )
        ]
        )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=output)  # 生成html文件并打开


if __name__ == '__main__':
    stations, underground_lines = build_data()
    plot_underground_lines('visualization_underground/London_railway.html', stations, underground_lines)
