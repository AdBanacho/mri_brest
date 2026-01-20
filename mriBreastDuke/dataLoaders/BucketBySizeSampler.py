import numpy as np
from torch.utils.data import Sampler

class BucketBySizeSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # Convert shape into a single scalar: volume = D*H*W
        self.keys = [np.prod(s) for s in dataset.sizes]

        # Sort indices by size
        self.indices = np.argsort(self.keys)

    def __iter__(self):
        # Yield sorted indices in batches
        batch = []
        for idx in self.indices:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1