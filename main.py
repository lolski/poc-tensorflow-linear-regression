import csv
import functools

import plotly
import plotly.graph_objs as g

def read_csv_file(path, skip_first_line):
    with open(path) as file:
        reader = csv.reader(file)

        if skip_first_line:
            next(reader, None)

        accumulator = []
        for row in reader:
            accumulator.append(row)

        return accumulator


def generate_scatter_plot(data):
    plot_x = list(map(lambda e: e[0], data))
    plot_y = list(map(lambda e: e[1], data))
    plotly.offline.plot({
        'data': [g.Scatter(x=plot_x, y=plot_y, mode='markers')]
    })


data = read_csv_file('res/train.csv', True)
generate_scatter_plot(data)