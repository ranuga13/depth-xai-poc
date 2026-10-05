def build_result_record(
    image_id,
    file_name,
    target_ann_id,
    target_class,
    occluder_ann_id,
    occluder_class,
    occlusion_ratio,
    role,
    perturbation_type,
    original_match_iou,
    metrics
):
    return {
        "image_id": image_id,
        "file_name": file_name,
        "target_ann_id": target_ann_id,
        "target_class": target_class,
        "occluder_ann_id": occluder_ann_id,
        "occluder_class": occluder_class,
        "occlusion_ratio": occlusion_ratio,
        "role": role,
        "perturbation_type": perturbation_type,
        "original_confidence": metrics["original_confidence"],
        "perturbed_confidence": metrics["perturbed_confidence"],
        "confidence_change": metrics["confidence_change"],
        "bbox_iou": metrics["bbox_iou"],
        "survived": metrics["survived"],
        "original_match_iou": original_match_iou
    }