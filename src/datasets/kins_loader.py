import json


class KINSLoader:
    def __init__(self, annotation_path):
        self.annotation_path = annotation_path

        with open(annotation_path, "r") as f:
            self.data = json.load(f)

        self.images_by_id = {
            image["id"]: image
            for image in self.data["images"]
        }

        self.categories_by_id = {
            category["id"]: category["name"]
            for category in self.data["categories"]
        }

        self.annotations_by_image = {}

        for ann in self.data["annotations"]:
            image_id = ann["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(ann)

    def get_image(self, image_id):
        return self.images_by_id.get(image_id)

    def get_annotations_for_image(self, image_id):
        return self.annotations_by_image.get(image_id, [])

    def get_category_name(self, category_id):
        return self.categories_by_id.get(category_id)

    def get_annotation_by_id(self, annotation_id):
        for ann in self.data["annotations"]:
            if ann["id"] == annotation_id:
                return ann

        return None

    def calculate_occlusion_ratio(self, annotation):
        a_area = annotation["a_area"]
        i_area = annotation["i_area"]

        if a_area <= 0:
            return 0.0

        return 1 - (i_area / a_area)