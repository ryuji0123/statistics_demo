from os.path import exists
from src.gauss_pdf_and_cdf import (GaussianProbabilisticSetting,
                                   BetaProbabilisticSetting)
import cv2


def show_or_save_figure(distribution_type):
    if distribution_type == "gaussian":
        distribution = GaussianProbabilisticSetting()
    elif distribution_type == "beta":
        distribution = BetaProbabilisticSetting()

    lengths = distribution.show_figure()

    for length in lengths:
        assert length[0] == length[1], 'length should be same'

    fig_path_all = distribution.save_figure()

    for fig_path in fig_path_all:
        assert exists(fig_path), 'file does not exist'
        assert fig_path.split('.')[-1] == 'png',\
               'file extension should be "png"'
        fig = cv2.imread(fig_path)
        assert fig is not None, 'figure is None'


def test_figure():
    show_or_save_figure("gaussian")
    show_or_save_figure("beta")
