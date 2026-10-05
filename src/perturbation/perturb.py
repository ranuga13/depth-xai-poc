import cv2


def mask_region(image, mask, value=0):
    perturbed = image.copy()

    perturbed[mask] = value

    return perturbed


def blur_region(
    image,
    mask,
    kernel_size=(31, 31)
):
    blurred = cv2.GaussianBlur(
        image,
        kernel_size,
        0
    )

    perturbed = image.copy()

    perturbed[mask] = blurred[mask]

    return perturbed