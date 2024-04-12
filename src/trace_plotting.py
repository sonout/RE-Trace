import folium
import numpy as np
import matplotlib.pylab as pl


def plot_trace(trace, is_sorted=False, color="red", my_map=None):
    # load map centered around mean
    if my_map is None:
        my_map = folium.Map(location=np.mean(trace, axis=0)[:2], zoom_start=1, tiles='cartodbpositronnolabels')

    folium.PolyLine(trace, color=color, weight=2.5, opacity=1).add_to(my_map)
    
    return my_map


def plot_two_traces(trace_1, trace_2, is_sorted=True):
    my_map = plot_trace(trace_1)
    my_map = plot_trace(trace_2, is_sorted=is_sorted, color="blue", my_map=my_map)
    return my_map



def plot_trace_w_points(trace, color="red", radius=3, my_map=None):
    # load map centered around mean
    if my_map is None:
        my_map = folium.Map(location=np.mean(trace, axis=0), zoom_start=15, tiles='cartodbpositronnolabels')

    folium.PolyLine(trace, color=color, weight=2.5, opacity=1).add_to(my_map)

    for p in trace:
        folium.CircleMarker(p[:2], color=color,
                            radius=radius,
                            weight=5).add_to(my_map)

    return my_map

def plot_two_traces_w_points(trace_1, trace_2, radius_1 = 2, radius_2 = 1, color_1 = 'red', color_2 = 'blue'):
    my_map = plot_trace_w_points(trace_1, color = color_1, radius= radius_1)
    my_map = plot_trace_w_points(trace_2, color =color_2, radius= radius_2, my_map=my_map)
    return my_map