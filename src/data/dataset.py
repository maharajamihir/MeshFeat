import torch
from torch.utils.data import Dataset, DataLoader

class MeshFeatDataset(Dataset):
    def __init__(self, x,triangles, y, points_xyz=None):
        self.x = x
        self.y = y
        self.triangles = triangles
        self.points_xyz = points_xyz

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve a single sample (x, y) from the dataset
        sample = {
            'x': self.x[idx],
            'y': self.y[idx],
            'triangle': self.triangles[idx]  # Add the triangle information to the sample dictionary
        }
        if self.points_xyz is not None:
            sample['xyz'] = self.points_xyz[idx]
        return sample

class MeshFeatDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = len(dataset)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size

        self.i = 0
        self.idxs = torch.arange(self.n_samples)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(self.n_samples)
        self.i = 0
        return self

    def _get_next_batch_idxs(self):
        low = self.i * self.batch_size
        high = min((self.i + 1) * self.batch_size, self.n_samples)
        self.i += 1
        return self.idxs[low:high]

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration

        batch_idxs = self._get_next_batch_idxs()
        batch = self.dataset.__getitem__(batch_idxs)

        return batch
