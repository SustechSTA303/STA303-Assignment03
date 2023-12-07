import plotly.graph_objects as go
import plotly.offline as py
from build_data import build_data

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def bernstein_poly(i, n, t):
    """
    计算 Bernstein 多项式的值
    """
    return comb(n, i) * (t**(n-i)) * ((1-t)**i)

def bezier_curve(control_points, t):
    """
    计算贝塞尔曲线上的点
    """
    n = len(control_points) - 1
    x = sum(bernstein_poly(i, n, t) * control_points[i][0] for i in range(n+1))
    y = sum(bernstein_poly(i, n, t) * control_points[i][1] for i in range(n+1))
    return x, y

def bezier_interpolation(points, num_points=100):
    """
    对一系列二维点进行贝塞尔曲线插值
    """
    t_values = np.linspace(0, 1, num_points)
    control_points = np.array(points)

    # 计算贝塞尔曲线上的点
    curve_points = np.array([bezier_curve(control_points, t) for t in t_values])

    return curve_points

# input_points = [(1, 2), (2, 5), (4, 6), (7, 3)]

# curve_points = bezier_interpolation(input_points)

# plt.scatter(*zip(*input_points), color='red', label='Original Points')
# plt.plot(*zip(*curve_points.T), label='Bezier Curve', linestyle='dashed', color='blue')
# plt.legend()
# plt.show()


def plot_path(path, output, stations, underground_lines):
    """
    :param path: A list of station name
    :param output: Path to output HTML
    :param stations: A mapping between station names and station objects of the name
    :param underground_lines: A mapping between underground lines name and a dictionary containing relevant
                             information about underground lines
    :return: None
    """

    # 检测路径是否存在

    for i in range(1, len(path)):
        if stations[path[i]] not in stations[path[i - 1]].links:
            raise Exception("path is not exist")
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
    

    
    for underground_line in underground_lines.values():
        tmp_lat=[stations[station_name].position[0] for station_name in underground_line['stations']],  # 路线点经度
        tmp_lon=[stations[station_name].position[1] for station_name in underground_line['stations']], 
        tmp_lat = tmp_lat[0]
        tmp_lon = tmp_lon[0]
        points = [stations[station_name].position for station_name in underground_line['stations']]
        curve_points = bezier_interpolation(points)
        print(curve_points)

        points = np.array(curve_points)

        data.extend([

            # 地铁路线
            go.Scattermapbox(
                lat=points[:, 0],
                lon=points[:, 1],
                mode='lines',
                line=go.scattermapbox.Line(
                    width=2,
                    color="#" + underground_line["colour"]
                ),
                name=underground_line['name'],
                legendgroup=underground_line['name'],
                showlegend=False
            ),
#             go.Scattermapbox(
#                 lat=underground_line['lat'],
#                 lon=underground_line['lon'],
#                 mode='lines',
#                 # 设置路线的参数
#                 line=go.scattermapbox.Line(
#                     width = 2,
# #                     color='black'
#                     color = "#" + underground_line["colour"]
#                 ),
#                 name = underground_line['name'],  # 线路名称，显示在图例（legend）上
#                 legendgroup = underground_line['name'],
#                 showlegend = False
#             ),
            go.Scattermapbox(
                lat=[stations[station_name].position[0] for station_name in underground_line['stations']],  # 路线点经度
                lon=[stations[station_name].position[1] for station_name in underground_line['stations']],  # 路线点纬度
                mode='markers',
                text=[stations[station_name].name for station_name in underground_line['stations']],
                # 设置标记点的参数
                marker=go.scattermapbox.Marker(
                    size=6,
#                     color='black'
                    color = "#" + underground_line["colour"]
                ),
                name=underground_line['name'],
                legendgroup=underground_line['name'],  # 设置与路线同组，当隐藏该路线时隐藏标记点
                showlegend=False  # 不显示图例（legend)
            )
        ])
    
    data.append(go.Scattermapbox(
        lat=[stations[station_name].position[0] for station_name in path],
        lon=[stations[station_name].position[1] for station_name in path],
        mode='markers+lines',
        text=path,
        line=go.scattermapbox.Line(
            width=3,
            color='rgba(255, 0, 0, 0.8)',
        ),
        marker=go.scattermapbox.Marker(
            size=8,
            color='red',
        ),
        name='my path'
    ))

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=output)  # 生成html文件并打开


if __name__ == '__main__':
    stations, underground_lines = build_data()
    plot_path(['Acton Town', 'Chiswick Park', 'Turnham Green', 'Stamford Brook'],
              'visualization_underground/my_path_in_London_railway.html', stations, underground_lines)
