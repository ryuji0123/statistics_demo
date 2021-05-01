from os import makedirs
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
    def visualize(x, y, title, fig_processing) -> str:
        '''Show or save figure

        Args:
            x: np.ndarray
                data points
            y: np.ndarray
                value of probability
            title: str
                figure title
            fig_processing: str
                show or save
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=title))
        fig.update_xaxes(title="x")
        fig.update_yaxes(title="y")
        fig.update_layout(title=title)
        if fig_processing == 'show':
            fig.show()
            return len(x), len(y)
        elif fig_processing == 'save':
            makedirs(FIGS_ROOT, exist_ok=True)
            fig_path = f'{FIGS_ROOT}/{title}.png'
            fig.write_image(fig_path)
            return fig_path


class GaussianProbabilisticSetting(ProbabilisticDistribution):
    '''Set data points and compute probability for gaussian distribution
    '''

    def __init__(self) -> None:
        '''Set data points and compute probability for gaussian distribution
        '''
        probability_x = np.linspace(-5, 5, 100)
        probability_y = []
        for i in range(len(probability_x)):
            probability_y.append(float(Distribution.gauss_np(probability_x[i])))
        probability_y = np.array(probability_y)

        self.probability = {"x": probability_x, "y": probability_y}

        x = Symbol('x')
        t = Symbol('t')

        # integrate gaussian pdf
        F = integrate(Distribution.gauss_sp(t), (t, -oo, x))  # gaussian cumulative density function (gaussian cdf)

        distribution_y = []
        t = np.linspace(-5, 5, 100)
        for i in range(len(t)):
            distribution_y.append(float(F.subs({x: t[i]})))
        distribution_x = probability_x
        distribution_y = np.array(distribution_y)
        self.distribution = {"x": distribution_x, "y": distribution_y}

    def show_figure(self) -> None:
        '''Show gaussian distribution figure
        '''
        length_probability = Visualizer.visualize(
            x=self.probability["x"],
            y=self.probability["y"],
            title="gaussian_pdf",
            fig_processing='show'
        )
        length_distribution = Visualizer.visualize(
            x=self.distribution["x"],
            y=self.distribution["y"],
            title="gaussian_cdf",
            fig_processing='show'
        )
        return length_probability, length_distribution

    def save_figure(self) -> str:
        '''Save gaussian distribution figure
        '''
        probability_fig_path = Visualizer.visualize(
            x=self.probability['x'],
            y=self.probability['y'],
            title='gaussian_pdf',
            fig_processing='save'
        )
        distribution_fig_path = Visualizer.visualize(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='gaussian_cdf',
            fig_processing='save'
        )
        return probability_fig_path, distribution_fig_path


class BetaProbabilisticSetting(ProbabilisticDistribution):
    '''Setting for beta distribution
    '''

    def __init__(self) -> None:
        '''Set data points and compute probability for beta distribution
        '''
        probability_x = np.linspace(0, 1, 100)
        a = 0.5
        b = 0.5
        probability_y = []
        for i in range(len(probability_x)):
            probability_y.append(float(Distribution.beta_pdf(probability_x[i], a, b)))
        probability_y = np.array(probability_y)

        self.probability = {'x': probability_x, 'y': probability_y}

        distribution_x = probability_x
        distribution_y = stats.beta.cdf(distribution_x, a, b)  # beta cumulative distribution function (beta cdf)
        self.distribution = {'x': distribution_x, 'y': distribution_y}

    def show_figure(self) -> None:
        '''Show beta distribution figure
        '''
        probability_fig_path = Visualizer.visualize(
            x=self.probability['x'],
            y=self.probability['y'],
            title='beta_pdf',
            fig_processing='show'
        )
        distribution_fig_path = Visualizer.visualize(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='beta_cdf',
            fig_processing='show'
        )
        return probability_fig_path, distribution_fig_path

    def save_figure(self) -> None:
        '''Save beta distribution figure
        '''
        probability_fig_path = Visualizer.visualize(
            x=self.probability['x'],
            y=self.probability['y'],
            title='beta_pdf',
            fig_processing='save'
        )
        distribution_fig_path = Visualizer.visualize(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='beta_cdf',
            fig_processing='save'
        )
        return probability_fig_path, distribution_fig_path


class Distribution:

    @staticmethod
    def gauss_np(t: float, mu: float = 0, sigma: float = 1) -> float:
        '''Compute gaussian probability density function.

        Args:
            t: data point
            mu: mean of gaussian probability density function
            sigma: standard deviation of gaussian probability density function
        Returns:
            gaussian probability density function
        '''
        return np.exp(- (t - mu) ** 2 / 2*sigma ** 2) / np.sqrt(2*np.pi*(sigma ** 2))

    @staticmethod
    def gauss_sp(t, mu: float = 0, sigma: float = 1) -> Symbol:
        '''Compute gaussian probability density function.

        Args:
            t: sympy.Symbol
                data point
            mu: mean of gaussian probability density function
            sigma: standard deviation of gaussian probability density function
        Returns:
            gaussian probability density function as sympy.Symbol
        '''
        return E ** (- (t - mu) ** 2 / 2*sigma ** 2) / sqrt(2*pi*(sigma ** 2))

    @staticmethod
    def beta_pdf(t: float, a: float = 0.5, b: float = 0.5) -> float:
        '''Compute beta probability density function.

        Args:
            t: data point
            mu: mean of beta probability density function
            sigma: standard deviation of beta probability density function
        Returns:
            beta probability density function
        '''
        B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        return t ** (a - 1) * (1 - t) ** (b - 1) / B


def main(distribution_type: str) -> None:
    if distribution_type == 'gaussian':
        distribution = GaussianProbabilisticSetting()
    elif distribution_type == 'beta':
        distribution = BetaProbabilisticSetting()
    distribution.show_figure()
    distribution.save_figure()


if __name__ == '__main__':
    main('gaussian')  # "gaussian" or "beta"
