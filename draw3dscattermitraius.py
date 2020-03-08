import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go

data = pd.read_csv('cc3d.csv')
datasize = data["delimeter"] * (10 ** 3)
datacolor = data["delimeter"]
fig1 = go.Scatter3d(x=data["MeanX"],
                    y=data["MeanY"],
                    z=data["MeanZ"],
                    marker=dict(opacity=0.9,
                                reversescale=True,

                                color=datacolor,
                                size=datasize),
                    line=dict(width=0.02),
                    mode='markers')
mylayout = go.Layout(scene=dict(xaxis=dict(title="x"),
                                yaxis=dict(title="y"),
                                zaxis=dict(title="z")), )
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                    auto_open=True,
                    filename=("3DPlot.html"))
