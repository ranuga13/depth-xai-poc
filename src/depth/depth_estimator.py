import numpy as np
import torch

from PIL import Image
from transformers import pipeline


class DepthEstimator:

    def __init__(
        self,
        model_name="depth-anything/Depth-Anything-V2-Small-hf"
    ):

        if torch.cuda.is_available():
            device = 0
        else:
            device = -1

        self.pipeline = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device
        )


    def predict(self, image_bgr):

        image_rgb = image_bgr[:, :, ::-1]

        pil_image = Image.fromarray(
            image_rgb
        )

        result = self.pipeline(
            pil_image
        )

        predicted_depth = result[
            "predicted_depth"
        ]

        if hasattr(
            predicted_depth,
            "detach"
        ):
            predicted_depth = (
                predicted_depth
                .detach()
                .cpu()
                .numpy()
            )

        predicted_depth = np.squeeze(
            predicted_depth
        )

        return predicted_depth