from sklearn.datasets import fetch_20newsgroups
from dataset import Dataset


class Dataset20Newsgroup(Dataset):

    def __init__(self, cfg=None):
        super().__init__()
        self.__dataset__ = fetch_20newsgroups(subset=cfg['subset'], categories=cfg['categories'],
                                              shuffle=cfg['shuffle'], random_state=cfg['random_state'])

