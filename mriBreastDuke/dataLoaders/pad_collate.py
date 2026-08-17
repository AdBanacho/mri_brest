import torch
import torch.nn.functional as F

def pad_collate(batch, max_series=5):
    """
    batch: list of (vol, label)
      vol: (K, 1, D, H, W), where K <= max_series

    Returns:
      vols:   (B, max_series, 1, Dmax, Hmax, Wmax)
      labels: (B,)
    """
    vols, labels = zip(*batch)

    fixed_vols = []
    for v in vols:
        v = v.float()

        # Ensure shape is (K, 1, D, H, W)
        if v.ndim == 4:          # (1, D, H, W) -> single series
            v = v.unsqueeze(0)   # (1, 1, D, H, W)

        assert v.ndim == 5, f"Expected 5D tensor, got {v.shape}"
        assert v.shape[1] == 1, f"Expected 1 channel, got {v.shape}"

        # Pad series dimension (K) to max_series
        k, c, d, h, w = v.shape
        if k > max_series:
            raise ValueError(
                f"Volume contains {k} series, more than max_series={max_series}."
            )
        if k < max_series:
            pad_k = max_series - k
            pad_tensor = torch.zeros(
                (pad_k, c, d, h, w),
                dtype=v.dtype,
                device=v.device,
            )
            v = torch.cat([v, pad_tensor], dim=0)

        fixed_vols.append(v)

    # Compute max spatial dimensions across batch
    max_d = max(v.shape[2] for v in fixed_vols)
    max_h = max(v.shape[3] for v in fixed_vols)
    max_w = max(v.shape[4] for v in fixed_vols)

    padded_vols = []
    for v in fixed_vols:
        k, c, d, h, w = v.shape

        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad last 3 dims: (W, H, D)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        v_padded = F.pad(v, padding, mode="constant", value=0.0)

        padded_vols.append(v_padded)

    # Stack into batch
    vols_tensor = torch.stack(padded_vols, dim=0)
    labels_tensor = torch.stack(labels, dim=0)

    vols_tensor = vols_tensor.squeeze(2) # (B, max_series, D, H, W)

    return vols_tensor, labels_tensor
