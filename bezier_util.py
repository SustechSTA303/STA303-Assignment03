import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.integrate import quad

def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(n-i)) * ((1-t)**i)

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    x = sum(bernstein_poly(i, n, t) * control_points[i][0] for i in range(n+1))
    y = sum(bernstein_poly(i, n, t) * control_points[i][1] for i in range(n+1))
    return x, y

def bezier_interpolation(points, num_points = 11):
    t_values = np.linspace(0, 1, num_points)
    control_points = np.array(points)

    curve_points = [bezier_curve(control_points, t) for t in t_values]

    return curve_points


def generate_control_point(pos1, pos2, pos3, pos4):
    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    x4, y4 = pos4
    l = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
    a = x2 - x1
    b = x3 - x4
    c = x3 - x2
    d = y2 - y1
    e = y3 - y4
    f = y3 - y2
    if b * d - a * e == 0:
        return (x2 + x3) / 2, (y2 + y3) / 2
    t = (b * f - e * c) / (b * d - a * e)
    s = (c * d - a * f) / (b * d - a * e)
    tx = t * a + x2
    ty = t * d + y2
    if t < 0 or s > 0:
        tx, ty = (x2 + x3) * 5 / 8 - (x1 + x4) * 1 / 8, (y2 + y3) * 5 / 8 - (y1 + y4) * 1 / 8
    return (tx, ty)

# def bezier_curve_quad(t, P1, P3, P2):
#     return tuple((1 - t) ** 2 * P1[i] + 2 * (1 - t) * t * P3[i] + t ** 2 * P2[i] for i in range(len(P1)))

def bezier_derivative(t, P1, P3, P2):
    return tuple(2 * (1 - t) * (P3[i] - P1[i]) + 2 * t * (P2[i] - P3[i]) for i in range(len(P1)))

def bezier_curve_length(P1, P3, P2, num_points=100):
    integrand = lambda t: np.linalg.norm(bezier_derivative(t, P1, P3, P2))
    length, error = quad(integrand, 0, 1)
    return length

def generate_bezier(underground_lines):
    cp_dict = {}
    for line_number, underground_line in underground_lines.items():
        tmp_lat=underground_line["lat"],  
        tmp_lon=underground_line["lon"], 
        tmp_lat = tmp_lat[0]
        tmp_lon = tmp_lon[0]
        points = [(tmp_lat[i], tmp_lon[i]) for i in range(len(tmp_lat))]
    
        line_dict = {}
        point_cnt = {}
        current_line = []

        for i in range(len(points)):
            if points[i] == (None, None):
                if points[i - 1] in line_dict:
                    line_dict[points[i - 1]].append(points[i - 2])
                else:
                    line_dict[points[i - 1]] = [points[i - 2]]
                    
                if points[i - 2] in line_dict:
                    line_dict[points[i - 2]].append(points[i - 1])
                else:
                    line_dict[points[i - 2]] = [points[i - 1]]
                    
            else:
                if points[i] in point_cnt:
                    point_cnt[points[i]] += 1
                else:
                    point_cnt[points[i]] = 1
        
        vis = set()
        curve_points = []
        while True:
            start_point = points[0]
            for k in point_cnt:
                if point_cnt[k] == 1 and k not in vis:
                    flg = 1
                    start_point = k
                    break
            
            if flg != 1:
                for k in point_cnt:
                    if len(line_dict[k]) > 0:
                        flg = 1
                        start_point = k
                        break
            
            if flg != 1:
                break
                
            curve_points.append(start_point)
            
            flg = 0
            ordered_points = []
        
            while start_point in line_dict:
                ordered_points.append(start_point)
                vis.add(start_point)
                point_cnt[start_point] -= 2
                tmp_lst = line_dict[start_point]
                tmp = start_point
                if len(tmp_lst) == 0:
                    break
                start_point = tmp_lst[0]
                for p in tmp_lst:
                    if point_cnt[p] >= 1:
                        start_point = p
                        break
                    elif p not in ordered_points:
                        start_point = p
                line_dict[tmp].remove(start_point)
                line_dict[start_point].remove(tmp)

            for i in range(1, len(ordered_points) - 2):
                if ordered_points[i] == ordered_points[i + 1]:
                    continue
                gcp = generate_control_point(ordered_points[i - 1], ordered_points[i], ordered_points[i + 1], ordered_points[i + 2])
                cp_dict[(line_number, ordered_points[i], ordered_points[i + 1])] = gcp
                cp_dict[(line_number, ordered_points[i + 1], ordered_points[i])] = gcp
                
    return cp_dict