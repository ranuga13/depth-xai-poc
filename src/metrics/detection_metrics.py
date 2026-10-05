from src.detector.target_matcher import calculate_iou


def calculate_detection_metrics(
    original_detection,
    perturbed_detection
):
    """
    Compare the original target detection with the
    matched detection after perturbation.
    """

    original_confidence = original_detection["confidence"]

    # Target was not matched after perturbation
    if perturbed_detection is None:
        return {
            "original_confidence": original_confidence,
            "perturbed_confidence": None,
            "confidence_change": None,
            "bbox_iou": None,
            "survived": False
        }

    perturbed_confidence = perturbed_detection["confidence"]

    confidence_change = (
        original_confidence
        - perturbed_confidence
    )

    bbox_iou = calculate_iou(
        original_detection["bbox"],
        perturbed_detection["bbox"]
    )

    return {
        "original_confidence": original_confidence,
        "perturbed_confidence": perturbed_confidence,
        "confidence_change": confidence_change,
        "bbox_iou": bbox_iou,
        "survived": True
    }