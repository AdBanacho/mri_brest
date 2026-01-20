import torch
import torch.nn.functional as F

def pad_collate(batch):
    """
    batch: list of (vol, label)
      vol: (1, D, H, W)
    Returns:
      vols: (B, 1, Dmax, Hmax, Wmax)
      labels: (B,)
    """
    vols, labels = zip(*batch)  # list of tensors, list of tensors

    # Ensure shape is (1, D, H, W)
    fixed_vols = []
    for v in vols:
        v = v.float()
        if v.ndim == 3:
            v = v.unsqueeze(0)  # (D, H, W) -> (1, D, H, W)
        assert v.ndim == 4, f"Expected 4D tensor, got {v.shape}"
        assert v.shape[0] == 1, f"Expected 1 channel, got {v.shape}"
        fixed_vols.append(v)

    # Compute max spatial dims in this batch
    max_d = max(v.shape[1] for v in fixed_vols)
    max_h = max(v.shape[2] for v in fixed_vols)
    max_w = max(v.shape[3] for v in fixed_vols)

    padded_vols = []
    for v in fixed_vols:
        _, d, h, w = v.shape
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w

        # F.pad: (w_left, w_right, h_left, h_right, d_left, d_right)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        v_padded = F.pad(v, padding, mode="constant", value=0.0)
        padded_vols.append(v_padded)

    # Stack along batch dimension → (B, 1, Dmax, Hmax, Wmax)
    vols_tensor = torch.stack(padded_vols, dim=0)

    labels_tensor = torch.stack(labels, dim=0)  # (B,)
    return vols_tensor, labels_tensor