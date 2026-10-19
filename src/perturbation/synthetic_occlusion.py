import numpy as np


def create_synthetic_occlusion_mask(
    target_annotation,
    height,
    width,
    occlusion_fraction
):
    """
    Create a rectangular synthetic occluder over the target.

    occlusion_fraction:
        0.10 = 10%
        0.30 = 30%
        0.50 = 50%
        0.70 = 70%
    """

    x, y, w, h = target_annotation["a_bbox"]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    target_width = max(1, x2 - x1)
    target_height = max(1, y2 - y1)

    occluder_width = int(
        target_width * occlusion_fraction
    )

    occluder_width = max(
        1,
        min(
            occluder_width,
            target_width
        )
    )

    mask = np.zeros(
        (height, width),
        dtype=bool
    )

    # Cover target from the left side toward the right
    occ_x1 = x1
    occ_x2 = min(
        x2,
        x1 + occluder_width
    )

    mask[
        y1:y2,
        occ_x1:occ_x2
    ] = True

    return mask