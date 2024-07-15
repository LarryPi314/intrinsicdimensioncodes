
import skdim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import make_swiss_roll

def generate_data(dist, n, d):
    if dist == "sphere":
        X = np.random.randn(n, d)
        X /= np.linalg.norm(X, axis=1)[:, None]
    elif dist == "swiss-roll":
        X, t = make_swiss_roll(n_samples=n, noise=0.05, random_state=0)
    elif dist == "uniform":
        X = np.random.rand(n, d)
    elif dist == "Gaussian":
        X = np.random.randn(n, d)
    elif dist == "hyperTwinPeaks":
        X = skdim.datasets.hyperTwinPeaks(n, d)
    elif dist == "hyperBall":
        X = skdim.datasets.hyperBall(n, d)
    elif dist == "benchmark-manifolds":
        X = skdim.datasets.BenchmarkManifolds()
    return X

def plot_subplots(data_list):
    fig = make_subplots(rows=len(data_list), cols=1, specs=[[{'type': 'Scatter3d'}] for _ in range(len(data_list))], subplot_titles=[f"Data batch {i}" for i in range(len(data_list))])
    traceList = []
    for ind, data in enumerate(data_list):
        trace = go.Scatter3d(dict(zip(['x','y','z'],data.T[:3])),
        mode='markers',marker=dict(size=1.5,colorbar=dict()))
        traceList.append(trace)
        fig.add_trace(trace, row=ind+1, col=1)
    fig.layout.update(height=800, width=800)
    fig.show()
    return

def twoNN_estimate(data):
    return skdim.id.TwoNN().fit(X = data).dimension_

def kNN_estimate(data):
    return skdim.id.KNN().fit(X = data).dimension_

def MLE_estimate(data, n_neighbors):
    '''
    Estimates the intrinsic dimension of the data locally assuming the data comes 
    from one manifold. This method is adopted from Levina, E., & Bickel, P. J. (2005).
    '''
    return skdim.id.MLE(K=n_neighbors).fit(X = data).dimension_

if __name__ == "__main__":
    data_list = []

    data1 = generate_data("sphere", 1000, 3)
    data2 = generate_data("swiss-roll", 1000, 3)
    data3 = generate_data("hyperTwinPeaks", 1000, 2)
    data_list.append(data1)
    data_list.append(data2)
    data_list.append(data3)

    plot_subplots(data_list)

    print("Starting twoNN estimate")
    for ind, data in enumerate(data_list):
        print(f"Estimate for data batch {ind}: {twoNN_estimate(data)}")

    print("Starting kNN estimate")
    for ind, data in enumerate(data_list):
        print(f"Estimate for data batch {ind}: {kNN_estimate(data)}")

    print("Starting MLE estimate")
    for ind, data in enumerate(data_list):
        print(f"Estimate for data batch {ind}: {MLE_estimate(data, 3)}")