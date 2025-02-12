import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class MyCobotInputs(transforms.DataTransformFn):
    """Inputs for the MyCobot policy.
    
    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [6] (6 joints for MyCobot 280pi)
    - actions: [action_horizon, 6]
    """
    action_dim: int
    adapt_to_pi: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_main",)

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 6 to the model action dim
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume base image always exists
        base_image = in_images["cam_main"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class MyCobotOutputs(transforms.DataTransformFn):
    """Outputs for the MyCobot policy."""
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 6 dims (6 joints)
        actions = np.asarray(data["actions"][:, :6])
        return {"actions": actions}

def make_mycobot_example() -> dict:
    """Creates a random input example for the MyCobot policy."""
    return {
        "state": np.ones((6,)),
        "images": {
            "cam_main": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }