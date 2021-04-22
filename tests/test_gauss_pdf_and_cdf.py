from os.path import exists
from src.gauss_pdf_and_cdf import (GaussianProbabilisticSetting,
                                   BetaProbabilisticSetting)
import cv2


def test_figure(distribution_type="gaussian"):
    if distribution_type == "gaussian":
        distribution = GaussianProbabilisticSetting()
    elif distribution_type == "beta":
        distribution = BetaProbabilisticSetting()

    distribution.show_figure()
    fig_path_all = distribution.save_figure()

    for fig_path in fig_path_all:
        assert exists(fig_path), 'file does not exist'
        assert fig_path.split('.')[-1] == 'png',\
               'file extension should be "png"'
        fig = cv2.imread(fig_path)
        assert fig is not None, 'figure is None'
