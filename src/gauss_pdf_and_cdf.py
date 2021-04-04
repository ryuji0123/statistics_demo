import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, pi, E, sqrt, integrate, oo
from abc import ABC, abstractmethod
from scipy import stats
import math


class ProbabilisticFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass


class Visualizer(ProbabilisticFunction):
    @staticmethod
    def visualize(x, y, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=title))
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="y")
        fig.update_layout(title=title)
        fig.show()


class GaussianProbabilisticSetting(ProbabilisticFunction):
    def __init__(self):
        pro_x = np.linspace(-5, 5, 100)
        pro_y = []
        for i in range(len(pro_x)):
            pro_y.append(float(gauss_np(pro_x[i])))
        pro_y = np.array(pro_y)

        self.prob = {"x": pro_x, "y": pro_y}

        x = Symbol('x')
        t = Symbol('t')

        # integrate gaussian pdf
        F = integrate(gauss_sp(t), (t, -oo, x))  # gaussian cumulative density function (gaussian cdf)

        dis_y = []
        t = np.linspace(-5, 5, 100)
        for i in range(len(t)):
            dis_y.append(float(F.subs({x: t[i]})))
        dis_x = pro_x
        dis_y = np.array(dis_y)
        self.dist = {"x": dis_x, "y": dis_y}

    def save_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="gaussian pdf")
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="gaussian cdf")


class BetaProbabilisticSetting(ProbabilisticFunction):
    def __init__(self):
        pro_x = np.linspace(0, 1, 100)

        pro_y = []
        for i in range(len(pro_x)):
            pro_y.append(float(beta_pdf(pro_x[i])))
        pro_y = np.array(pro_y)

        self.prob = {"x": pro_x, "y": pro_y}

        dis_x = pro_x
        a = 0.5
        b = 0.5
        dis_y = stats.beta.cdf(dis_x, a, b)  # beta cumulative distribution function (beta cdf)
        self.dist = {"x": dis_x, "y": dis_y}

    def save_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="beta pdf")
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="beta cdf")


def gauss_np(t, mu=0, sigma=1):
    return np.exp(-(t-mu) ** 2 / 2 * sigma ** 2) / np.sqrt(2 * np.pi * (sigma ** 2))


def gauss_sp(t, mu=0, sigma=1):
    return E**(-(t-mu) ** 2 / 2 * sigma ** 2) / sqrt(2 * pi * (sigma ** 2))


def beta_pdf(t, a=0.5, b=0.5):
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return t ** (a - 1) * (1 - t) ** (b - 1) / B


def main(dist_type):
    if dist_type == "gaussian":
        dist = GaussianProbabilisticSetting()
    if dist_type == "beta":
        dist = BetaProbabilisticSetting()
    dist.save_figure()


if __name__ == "__main__":
    main("beta")  # "gaussian" or "beta"
