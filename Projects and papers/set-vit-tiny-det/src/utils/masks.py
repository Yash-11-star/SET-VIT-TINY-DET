import torch

def boxes_to_mask(boxes_xyxy, H, W, device):
    """
    Rasterize axis-aligned boxes into a binary mask (1 on FG).
    boxes_xyxy: (N,4) in absolute image coords scaled to (H,W) already.
    """
    M = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        M[:, y1:y2, x1:x2] = 1.0
    return M

def downscale_boxes(boxes_xyxy, scale_y, scale_x):
    scaled = boxes_xyxy.clone()
    scaled[:, [0,2]] *= scale_x
    scaled[:, [1,3]] *= scale_y
    return scaled
