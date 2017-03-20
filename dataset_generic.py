from sklearn.datasets import load_files
from dataset import Dataset


class DatasetGeneric(Dataset):

    def __init__(self, container_path=None, categories=None, load_content=True,
                 encoding='utf-8', shuffle=True, random_state=42):
        """
        Load text files with categories as subfolder names.
        Individual samples are assumed to be files stored a two levels folder structure.
        :param container_path: The path of the container
        :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
        :param shuffle: shuffle the list or not
        :param random_state: seed integer to shuffle the dataset
        :return: data and labels of the dataset
        """
        super().__init__()
        self.__dataset__ = load_files(container_path=container_path, categories=categories,
                                  load_content=load_content, shuffle=shuffle, encoding=encoding,
                                  random_state=random_state)
