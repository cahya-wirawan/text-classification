from sklearn.datasets import fetch_20newsgroups
from dataset import Dataset


class Dataset20newsgroup(Dataset):

    def __init__(self, subset='train', categories=None, shuffle=True, random_state=42):
        super().__init__()
        self.__dataset__ = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
