##returns up to 10 cars that are 30–50% occluded and at least 70×50 pixels in their amodal bounding box.
from datasets.kins_loader import KINSLoader
from datasets.sample_filter import find_suitable_targets


ANNOTATION_PATH = "data/annotations/update_train_2020.json"


loader = KINSLoader(
    ANNOTATION_PATH
)


targets = find_suitable_targets(
    loader=loader,
    target_class="car",
    min_occlusion=0.20,
    max_occlusion=0.60,
    min_width=70,
    min_height=50,
    max_targets=10
)


print(
    f"Found {len(targets)} suitable targets"
)

print()


for item in targets:

    ann = item["annotation"]

    image = loader.get_image(
        ann["image_id"]
    )

    print(
        f"Image: {image['file_name']} | "
        f"Ann ID: {ann['id']} | "
        f"Class: {item['class_name']} | "
        f"Occlusion: "
        f"{item['occlusion_ratio']:.3f} | "
        f"BBox: {ann['a_bbox']}"
    )