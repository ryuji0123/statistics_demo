from os import makedirs
from os.path import join
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, pi, E, sqrt, integrate, oo
from abc import ABC, abstractmethod
from scipy import stats
import math
from config import FIGS_ROOT


class ProbabilisticDistribution(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def show_figure(self):
        pass

    @abstractmethod
    def save_figure(self):
        pass


class Visualizer:

    @staticmethod
    def visualize(x, y, title, fig_processing):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=title))
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="y")
        fig.update_layout(title=title)
        if fig_processing == 'show':
            fig.show()
        elif fig_processing == 'save':
            makedirs(FIGS_ROOT, exist_ok=True)
            fig.write_image(join(FIGS_ROOT, title+".png"))


class GaussianProbabilisticSetting(ProbabilisticDistribution):

    def __init__(self):
        pro_x = np.linspace(-5, 5, 100)
        pro_y = []
        for i in range(len(pro_x)):
            pro_y.append(float(Distribution.gauss_np(pro_x[i])))
        pro_y = np.array(pro_y)

        self.prob = {"x": pro_x, "y": pro_y}

        x = Symbol('x')
        t = Symbol('t')

        # integrate gaussian pdf
        F = integrate(Distribution.gauss_sp(t), (t, -oo, x))  # gaussian cumulative density function (gaussian cdf)

        dis_y = []
        t = np.linspace(-5, 5, 100)
        for i in range(len(t)):
            dis_y.append(float(F.subs({x: t[i]})))
        dis_x = pro_x
        dis_y = np.array(dis_y)
        self.dist = {"x": dis_x, "y": dis_y}

    def show_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="gaussian_pdf", fig_processing='show')
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="gaussian_cdf", fig_processing='show')

    def save_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="gaussian_pdf", fig_processing='save')
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="gaussian_cdf", fig_processing='save')


class BetaProbabilisticSetting(ProbabilisticDistribution):

    def __init__(self):
        pro_x = np.linspace(0, 1, 100)
        a = 0.5
        b = 0.5
        pro_y = []
        for i in range(len(pro_x)):
            pro_y.append(float(Distribution.beta_pdf(pro_x[i], a, b)))
        pro_y = np.array(pro_y)

        self.prob = {"x": pro_x, "y": pro_y}

        dis_x = pro_x
        dis_y = stats.beta.cdf(dis_x, a, b)  # beta cumulative distribution function (beta cdf)
        self.dist = {"x": dis_x, "y": dis_y}

    def show_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="beta_pdf", fig_processing='show')
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="beta_cdf", fig_processing='show')

    def save_figure(self):
        Visualizer.visualize(x=self.prob["x"], y=self.prob["y"], title="beta_pdf", fig_processing='save')
        Visualizer.visualize(x=self.dist["x"], y=self.dist["y"], title="beta_cdf", fig_processing='save')


class Distribution:

    @staticmethod
    def gauss_np(t, mu=0, sigma=1):
        return np.exp(-(t-mu) ** 2 / 2 * sigma ** 2) / np.sqrt(2 * np.pi * (sigma ** 2))

    @staticmethod
    def gauss_sp(t, mu=0, sigma=1):
        return E**(-(t-mu) ** 2 / 2 * sigma ** 2) / sqrt(2 * pi * (sigma ** 2))

    @staticmethod
    def beta_pdf(t, a=0.5, b=0.5):
        B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        return t ** (a - 1) * (1 - t) ** (b - 1) / B


def main(dist_type):
    if dist_type == "gaussian":
        dist = GaussianProbabilisticSetting()
    if dist_type == "beta":
        dist = BetaProbabilisticSetting()
    dist.show_figure()
    dist.save_figure()


if __name__ == "__main__":
    main("gaussian")  # "gaussian" or "beta"
