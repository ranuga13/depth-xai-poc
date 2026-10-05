import cv2
import matplotlib.pyplot as plt

from datasets.kins_loader import KINSLoader
from roles.scene_roles import build_scene_roles
from perturbation.perturb import mask_region, blur_region


ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_PATH = "data/images/training/image_2/006090.png"

TARGET_ANN_ID = 68


loader = KINSLoader(
    ANNOTATION_PATH
)

target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

image_annotations = loader.get_annotations_for_image(
    target_ann["image_id"]
)

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

roles = build_scene_roles(
    target_annotation=target_ann,
    image_annotations=image_annotations,
    height=height,
    width=width
)


target_masked = mask_region(
    image,
    roles["target_mask"]
)

target_blurred = blur_region(
    image,
    roles["target_mask"]
)


image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

masked_rgb = cv2.cvtColor(
    target_masked,
    cv2.COLOR_BGR2RGB
)

blurred_rgb = cv2.cvtColor(
    target_blurred,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(masked_rgb)
plt.title("Target Masked")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blurred_rgb)
plt.title("Target Blurred")
plt.axis("off")

plt.tight_layout()
plt.show()