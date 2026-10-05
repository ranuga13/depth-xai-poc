import cv2
import matplotlib.pyplot as plt

from datasets.kins_loader import KINSLoader
from roles.scene_roles import build_scene_roles


ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_PATH = "data/images/training/image_2/006090.png"

TARGET_ANN_ID = 68


loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

image_annotations = (
    loader.get_annotations_for_image(
        target_ann["image_id"]
    )
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
    width=width,
    context_scale=1.5
)


print("Target annotation:")
print(target_ann["id"])

print()

if roles["occluder_annotation"] is not None:
    print(
        "Occluder annotation:",
        roles["occluder_annotation"]["id"]
    )
else:
    print("No occluder found")


plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(
    roles["target_mask"],
    cmap="gray"
)
plt.title("Target")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    roles["occluder_mask"],
    cmap="gray"
)
plt.title("Foreground Occluder")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(
    roles["context_mask"],
    cmap="gray"
)
plt.title("Local Context")
plt.axis("off")

plt.tight_layout()
plt.show()