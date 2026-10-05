from datasets.kins_loader import KINSLoader


ANNOTATION_PATH = "data/annotations/update_train_2020.json"


loader = KINSLoader(
    ANNOTATION_PATH
)


print("Number of images:")
print(len(loader.images_by_id))

print()

print("Categories:")
print(loader.categories_by_id)

print()


# Test image 7
image_id = 7

image = loader.get_image(image_id)

print("Image:")
print(image)

print()


annotations = loader.get_annotations_for_image(
    image_id
)

print(
    f"Number of annotations in image {image_id}: "
    f"{len(annotations)}"
)

print()


# Print first few annotations
for ann in annotations[:5]:

    category = loader.get_category_name(
        ann["category_id"]
    )

    occlusion_ratio = loader.calculate_occlusion_ratio(
        ann
    )

    print(
        f"Annotation ID: {ann['id']} | "
        f"Class: {category} | "
        f"Occlusion: {occlusion_ratio:.3f}"
    )