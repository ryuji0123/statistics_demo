from os.path import exists
from src.distribution_function import (
    GaussianProbabilisticDistributionHandler,
    BetaProbabilisticDistributionHandler
)
import cv2


def examine_show_or_save_figure_function(distribution_type: str) -> None:
    '''Test show_figure, save_figure function.

    Args:
        distribution_type: 'gaussian' or 'beta'
    '''
    if distribution_type == 'gaussian':
        distribution = GaussianProbabilisticDistributionHandler()
    elif distribution_type == 'beta':
        distribution = BetaProbabilisticDistributionHandler()
    else:
        raise NotImplementedError('Distriution type should be "gaussian" or "beta"')

    fig_path_all = distribution.save_figure()

    for fig_path in fig_path_all:
        assert exists(fig_path), 'File does not exist'
        assert fig_path.split('.')[-1] == 'png',\
               'File extension should be "png"'
        fig = cv2.imread(fig_path)
        assert fig is not None, 'Figure is None'


def test_figure():
    examine_show_or_save_figure_function('gaussian')
    examine_show_or_save_figure_function('beta')
