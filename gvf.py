import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd
import folder_paths


class CropToAspectRatioMinMax:
    """Crops an image to a max and min aspect ratio, only if such is needed"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", ),
                             "min_aspect": ("FLOAT", {"default": 0.6666666666666, "min": 0.1, "max": 10.0, "step": 0.01}),
                             "max_aspect": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "crop_to_aspect_min_max"
    CATEGORY = "gvf"

    def crop_to_aspect_min_max(self, images, min_aspect, max_aspect):
        # images: [B, H, W, C]
        _, h, w, _ = images.shape
        aspect = w / h

        if min_aspect <= aspect <= max_aspect:
            return (images, w, h)

        # Move channels to first for slicing: [B, C, H, W]
        images = images.movedim(-1, 1)

        if aspect > max_aspect:
            # Too wide.  crop width
            new_w = int(h * max_aspect)
            offset = (w - new_w) // 2
            images = images[:, :, :, offset:offset + new_w]
        else:
            # Too tall.  crop height
            new_h = int(w / min_aspect)
            offset = (h - new_h) // 2
            images = images[:, :, offset:offset + new_h, :]

        # Move channels back to last: [B, H, W, C]
        images = images.movedim(1, -1)
        return (images, w, h)


class CropToAspectRatio:
    """Crops an image to a specific apect ratio"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", ),
                             "target_aspect": ("FLOAT", {"default": 0.6666666666666, "min": 0.1, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "crop_to_aspect"
    CATEGORY = "gvf"

    def crop_to_aspect(self, images, target_aspect):
        # images: [B, H, W, C]
        _, h, w, _ = images.shape
        aspect = w / h

        if target_aspect == aspect:
            return (images, w, h)

        # Move channels to first for slicing: [B, C, H, W]
        images = images.movedim(-1, 1)

        if aspect > target_aspect:
            # Too wide.  crop width
            new_w = int(h * target_aspect)
            offset = (w - new_w) // 2
            images = images[:, :, :, offset:offset + new_w]
        else:
            # Too tall.  crop height
            new_h = int(w / target_aspect)
            offset = (h - new_h) // 2
            images = images[:, :, offset:offset + new_h, :]

        # Move channels back to last: [B, H, W, C]
        images = images.movedim(1, -1)
        return (images, w, h)


class IfElseFloat:
    """Return the first float if true, otherwise the second"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"if_true": ("FLOAT", {"default": 1.0}),
                "if_false": ("FLOAT", {"default": 0.0}),
                "boolean": ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "if_else_float"
    CATEGORY = "gvf"

    def if_else_float(self, if_true, if_false, boolean):
        return (if_true if boolean else if_false,)


class SizeFromAspect:
    """For a given length of the short side and aspect ratio, produce height and width"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                  "short_length": ("INT", {"default": 1024, "min": 256, "max": 4080, "step": 1}),
                  "aspect": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01})}}
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Width", "Height")
    FUNCTION = "size_from_aspect"
    CATEGORY = "gvf"

    def size_from_aspect(self, short_length, aspect):
        if aspect > 1.0:
            return (short_length, int(short_length * aspect))
        return (int(short_length * aspect), short_length)


class IfElseInt:
    """Return the first int if true, otherwise the second"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"if_true": ("INT", {"default": 1}),
                "if_false": ("INT", {"default": 0}),
                "boolean": ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "if_else_int"
    CATEGORY = "gvf"

    def if_else_int(self, if_true, if_false, boolean):
        return (if_true if boolean else if_false,)


class CheckpointLoaderWithName:
    """Load checkpoint but also return the checkpoint name"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "checkpoint name")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "Name of the model, as a string",
                       )
    FUNCTION = "load_checkpoint_with_name"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint_with_name(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return *out[:3], ckpt_name


NODE_CLASS_MAPPINGS = {
    "CropToAspectRatioMinMax": CropToAspectRatioMinMax,
    "CropToAspectRatio": CropToAspectRatio,
    "IfElseFloat": IfElseFloat,
    "SizeFromAspect": SizeFromAspect,
    "IfElseInt": IfElseInt,
    "CheckpointLoaderWithName": CheckpointLoaderWithName,
    # "StringContains": StringContains,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropToAspectRatioMinMax": "Crop Image to Min/Max Aspect Ratio",
    "CropToAspectRatio": "Crop Image to Aspect Ratio",
    "IfElseFloat": "If else with two float values",
    "IfElseInt": "If else with two int values",
    "SizeFromAspect": "Image size from aspect ratio",
    "CheckpointLoaderWithName": "Load checkpoint and provide its name as a string",
    # "StringContains": "Return if a string has a substring",
}
