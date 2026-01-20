from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler

class BucketByExactSizeBatchSampler(Sampler):
    """
    Yields batches (lists of indices) where every sample in the batch has EXACTLY the same size.
    Assumes dataset.sizes is a list/array of shape-like objects (e.g., (D,H,W)).
    """
    def __init__(self, dataset, batch_size, shuffle_within_size=True, drop_last=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_within_size = bool(shuffle_within_size)
        self.drop_last = bool(drop_last)

        # Group indices by exact size (tuple is hashable; list/np-array is not)
        buckets = defaultdict(list)
        for i, s in enumerate(dataset.sizes):
            key = tuple(s)  # exact match by (D,H,W) etc.
            buckets[key].append(i)

        self.buckets = dict(buckets)
        self.bucket_keys = sorted(self.buckets.keys(), key=lambda k: np.prod(k))

    def __iter__(self):
        rng = np.random.default_rng()

        # Iterate bucket-by-bucket; every yielded batch is uniform size
        for key in self.bucket_keys:
            idxs = self.buckets[key].copy()

            if self.shuffle_within_size:
                rng.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            limit = n_full * self.batch_size

            for start in range(0, limit, self.batch_size):
                yield idxs[start:start + self.batch_size]

            # leftovers (mixed sizes not allowed) -> either drop or yield smaller batch
            if not self.drop_last:
                rem = idxs[limit:]
                if rem:
                    yield rem

    def __len__(self):
        if self.drop_last:
            return sum(len(v) // self.batch_size for v in self.buckets.values())
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.buckets.values())
