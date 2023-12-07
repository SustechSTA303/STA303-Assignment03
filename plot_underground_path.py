import plotly.graph_objects as go
import plotly.offline as py
from build_data import build_data

from bezier_util import bezier_interpolation
import numpy as np

def plot_path(path, output, stations, underground_lines, cp_dict, metric):
    """
    :param path: A list of station name
    :param output: Path to output HTML
    :param stations: A mapping between station names and station objects of the name
    :param underground_lines: A mapping between underground lines name and a dictionary containing relevant
                             information about underground lines
    :return: None
    """

#     # 检测路径是否存在

#     for i in range(1, len(path)):
#         if stations[path[i]] not in stations[path[i - 1]].links:
#             raise Exception("path is not exist")
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
    
    for line_number, underground_line in underground_lines.items():
        tmp_lat=underground_line["lat"],  
        tmp_lon=underground_line["lon"], 
        tmp_lat = tmp_lat[0]
        tmp_lon = tmp_lon[0]
        points = [(tmp_lat[i], tmp_lon[i]) for i in range(len(tmp_lat))]
    
        line_dict = {}
        point_cnt = {}
        ordered_points = []

        for i in range(len(points)):
            if points[i] == (None, None):
                if metric == "Bezier" \
                        and (line_number, points[i - 1], points[i - 2]) in cp_dict:
                    control_points = [
                        points[i - 1], 
                        cp_dict[(line_number, points[i - 1], points[i - 2])], 
                        points[i - 2]
                    ]
                    ordered_points += bezier_interpolation(control_points)
                    ordered_points.append((None, None))
                elif metric == "Manhattan":
                    ordered_points += [
                        points[i - 1], 
                        (points[i - 1][0], points[i - 2][1]),
                        points[i - 2], 
                        (None, None)
                    ]
                else:
                    ordered_points += [points[i - 1], points[i - 2], (None, None)]
            
        points = np.array(ordered_points)

        data.extend([
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
                legendgroup=underground_line['name'],  
                showlegend=False  
            )
        ])
    points = []
    ordered_points = []

    for i in range(1, len(path)):
        p1 = stations[path[i - 1]]
        p2 = stations[path[i]]
        l = {s[1] for s in p1.links}.intersection({s[1] for s in p2.links})
        result = None
        for li in l:
            result = cp_dict.get((int(li), p1.position, p2.position))
            if result != None:
                break
        if metric == "Bezier" and result != None:
            control_points = [p2.position, result, p1.position]
            ordered_points += bezier_interpolation(control_points)
            ordered_points.append((None, None))
        elif metric == "Manhattan":
            ordered_points += [
                p1.position,
                (p1.position[0], p2.position[1]),
                p2.position, 
                (None, None)
            ]
        else:
            ordered_points += [p1.position, p2.position, (None, None)]  
    points = np.array(ordered_points)
    data.extend([
        go.Scattermapbox(
            lat=points[:, 0],
            lon=points[:, 1],
            mode='lines',
            line=go.scattermapbox.Line(
                width=3,
                color='rgba(255, 0, 0, 0.8)',
            ),
            name=f'Path {path[0]} -> {path[-1]}',
            showlegend=False
        ),
        go.Scattermapbox(
            lat=[stations[station_name].position[0] for station_name in path], 
            lon=[stations[station_name].position[1] for station_name in path], 
            mode='markers',
            text=path,
            marker=go.scattermapbox.Marker(
                size=8,
                color = 'rgba(255, 0, 0, 0.8)',
            ),
            name=f'Path {path[0]} -> {path[-1]}',
        )
    ])

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=output)  


if __name__ == '__main__':
    stations, underground_lines = build_data()
    plot_path(['Acton Town', 'Chiswick Park', 'Turnham Green', 'Stamford Brook'],
              'visualization_underground/my_path_in_London_railway.html', stations, underground_lines)
