from paddle.io import Dataset
from random import shuffle


class SimCSEUnsupervisedDataset(Dataset):
    def __init__(self, examples: [str]):
        super(SimCSEUnsupervisedDataset, self).__init__()
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index], 1 #1用来占位，是多余的

    def __len__(self):
        return len(self.examples)

    def shuffle(self):
        shuffle(self.examples)

