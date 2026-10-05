def find_suitable_targets(
    loader,
    target_class="car",
    min_occlusion=0.20,
    max_occlusion=0.60,
    min_width=100,
    min_height=60,
    min_area=6000,
    max_targets=None
):
    suitable = []

    for ann in loader.data["annotations"]:
        class_name = loader.get_category_name(
            ann["category_id"]
        )

        if class_name != target_class:
            continue

        occlusion_ratio = loader.calculate_occlusion_ratio(
            ann
        )

        if not (
            min_occlusion
            <= occlusion_ratio
            <= max_occlusion
        ):
            continue

        x, y, w, h = ann["a_bbox"]

        # Reject objects that are too small
        if w < min_width or h < min_height:
            continue

        # Reject objects with a small overall bounding-box area
        bbox_area = w * h

        if bbox_area < min_area:
            continue

        suitable.append(
            {
                "annotation": ann,
                "class_name": class_name,
                "occlusion_ratio": occlusion_ratio
            }
        )

        if (
            max_targets is not None
            and len(suitable) >= max_targets
        ):
            break

    return suitable