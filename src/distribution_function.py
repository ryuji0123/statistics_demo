from os import makedirs
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, pi, E, sqrt, integrate, oo
from abc import ABC, abstractmethod
from scipy import stats
import math
from config import FIGS_ROOT


class ProbabilisticDistributionHandler(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def show_figure(self):
        pass

    @abstractmethod
    def save_figure(self):
        pass


class Visualizer(object):

    def visualize(self, x: np.ndarray, y: np.ndarray, title: str) -> go.Figure:
        '''Show or save figure.

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
        fig.update_xaxes(title='x')
        fig.update_yaxes(title='y')
        fig.update_layout(title=title)
        return fig

    def show(self, x: np.ndarray, y: np.ndarray, title: str) -> None:
        fig = self.visualize(x, y, title)
        fig.show()

    def save(self, x: np.ndarray, y: np.ndarray, title: str) -> str:
        fig = self.visualize(x, y, title)
        makedirs(FIGS_ROOT, exist_ok=True)
        fig_path = f'{FIGS_ROOT}/{title}.png'
        fig.write_image(fig_path)
        return fig_path


class GaussianProbabilisticDistributionHandler(ProbabilisticDistributionHandler):
    '''Set data points and compute probability for gaussian distribution.
    '''

    def __init__(self) -> None:
        '''Set data points and compute probability for gaussian distribution.
        '''
        probability_x = np.linspace(-5, 5, 100)
        probability_y = []
        for i in range(len(probability_x)):
            probability_y.append(float(Distribution.gauss_np(probability_x[i])))

        probability_y = np.array(probability_y)
        assert len(probability_x) == len(probability_y), 'X and y must be the same size'

        self.probability = {'x': probability_x, 'y': probability_y}

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

        assert len(distribution_x) == len(distribution_y), 'X and y must be the same size'
        self.distribution = {'x': distribution_x, 'y': distribution_y}

        self.visualize_figure = Visualizer()

    def show_figure(self) -> None:
        '''Show gaussian distribution figure.
        '''
        self.visualize_figure.show(
            x=self.probability['x'],
            y=self.probability['y'],
            title='gaussian_pdf'
        )
        self.visualize_figure.show(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='gaussian_cdf'
        )

    def save_figure(self) -> str:
        '''Save gaussian distribution figure.
        '''
        probability_fig_path = self.visualize_figure.save(
            x=self.probability['x'],
            y=self.probability['y'],
            title='gaussian_pdf'
        )
        distribution_fig_path = self.visualize_figure.save(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='gaussian_cdf'
        )
        return probability_fig_path, distribution_fig_path


class BetaProbabilisticDistributionHandler(ProbabilisticDistributionHandler):
    '''Setting for beta distribution.
    '''

    def __init__(self, a: float = 0.5, b: float = 0.5) -> None:
        '''Set data points and compute probability for beta distribution.

        Args:
            a, b: parameters in beta function
        '''
        probability_x = np.linspace(0, 1, 100)
        probability_y = []
        for i in range(len(probability_x)):
            probability_y.append(float(Distribution.beta_pdf(probability_x[i], a, b)))
        probability_y = np.array(probability_y)

        assert len(probability_x) == len(probability_y), 'X and y must be the same size'
        self.probability = {'x': probability_x, 'y': probability_y}

        distribution_x = probability_x
        distribution_y = stats.beta.cdf(distribution_x, a, b)  # beta cumulative distribution function (beta cdf)

        assert len(distribution_x) == len(distribution_y), 'X and y must be the same size'
        self.distribution = {'x': distribution_x, 'y': distribution_y}

        self.visualize_figure = Visualizer()

    def show_figure(self) -> None:
        '''Show beta distribution figure
        '''
        self.visualize_figure.show(
            x=self.probability['x'],
            y=self.probability['y'],
            title='beta_pdf'
        )
        self.visualize_figure.show(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='beta_cdf'
        )

    def save_figure(self) -> str:
        '''Save beta distribution figure
        '''
        probability_fig_path = self.visualize_figure.save(
            x=self.probability['x'],
            y=self.probability['y'],
            title='beta_pdf'
        )
        distribution_fig_path = self.visualize_figure.save(
            x=self.distribution['x'],
            y=self.distribution['y'],
            title='beta_cdf'
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
        return np.exp(- (t - mu) ** 2 / 2 * sigma ** 2) / np.sqrt(2 * np.pi * (sigma ** 2))

    @staticmethod
    def gauss_sp(t: Symbol, mu: float = 0, sigma: float = 1) -> Symbol:
        '''Compute gaussian probability density function.

        Args:
            t: sympy.Symbol
                data point
            mu: mean of gaussian probability density function
            sigma: standard deviation of gaussian probability density function
        Returns:
            gaussian probability density function as sympy.Symbol
        '''
        return E ** (- (t - mu) ** 2 / 2 * sigma ** 2) / sqrt(2 * pi * (sigma ** 2))

    @staticmethod
    def beta_pdf(t: float, a: float = 0.5, b: float = 0.5) -> float:
        '''Compute beta probability density function.

        Args:
            t: data point
            a, b: parameters in beta function
        Returns:
            beta probability density function
        '''
        B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        return t ** (a - 1) * (1 - t) ** (b - 1) / B


def main(distribution_type: str) -> None:
    if distribution_type == 'gaussian':
        distribution = GaussianProbabilisticDistributionHandler()
    elif distribution_type == 'beta':
        distribution = BetaProbabilisticDistributionHandler()
    else:
        raise NotImplementedError('Distriution type should be "gaussian" or "beta".')
    distribution.show_figure()
    distribution.save_figure()


if __name__ == '__main__':
    main('gaussian')  # 'gaussian' or 'beta'
