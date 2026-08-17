import unittest

import torch

from .pad_collate import pad_collate


class PadCollateTest(unittest.TestCase):
    def test_pads_to_configured_series_count(self):
        batch = [
            (torch.ones(2, 1, 2, 3, 4), torch.tensor(0)),
            (torch.ones(4, 1, 1, 2, 3), torch.tensor(1)),
        ]

        volumes, labels = pad_collate(batch, max_series=4)

        self.assertEqual(tuple(volumes.shape), (2, 4, 2, 3, 4))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertTrue(torch.count_nonzero(volumes[0, 2:]) == 0)

    def test_rejects_more_series_than_configured(self):
        batch = [(torch.ones(5, 1, 2, 2, 2), torch.tensor(0))]

        with self.assertRaisesRegex(ValueError, "more than max_series=4"):
            pad_collate(batch, max_series=4)
