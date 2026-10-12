import random
import numpy as np


def create_random_local_mask(
    target_annotation,
    height,
    width,
    scale=1.5,
    mask_fraction=0.25
):
    x, y, w, h = target_annotation["a_bbox"]

    cx = x + w / 2
    cy = y + h / 2

    local_w = w * scale
    local_h = h * scale

    x1 = max(0, int(cx - local_w / 2))
    y1 = max(0, int(cy - local_h / 2))
    x2 = min(width, int(cx + local_w / 2))
    y2 = min(height, int(cy + local_h / 2))

    region_w = max(1, x2 - x1)
    region_h = max(1, y2 - y1)

    mask_w = max(
        1,
        int(region_w * mask_fraction)
    )

    mask_h = max(
        1,
        int(region_h * mask_fraction)
    )

    max_x = max(x1, x2 - mask_w)
    max_y = max(y1, y2 - mask_h)

    random_x = random.randint(
        x1,
        max_x
    )

    random_y = random.randint(
        y1,
        max_y
    )

    mask = np.zeros(
        (height, width),
        dtype=bool
    )

    mask[
        random_y:random_y + mask_h,
        random_x:random_x + mask_w
    ] = True

    return mask