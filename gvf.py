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


class IfElseValues:
    """Return the first value if true, otherwise the second"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"if_true": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "if_false": ("FLOAT", {"default": 0.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "condition": ("BOOL", {"default": False}),
                             }}
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "if_else_values"
    CATEGORY = "gvf"

    def if_else_values(self, if_true, if_false, condition):
        return (if_true if condition else if_false,)


NODE_CLASS_MAPPINGS = {
    "CropToAspectRatioMinMax": CropToAspectRatioMinMax,
    "CropToAspectRatio": CropToAspectRatio,
    "IfElseValues": IfElseValues,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropToAspectRatioMinMax": "Crop Image to Min/Max Aspect Ratio",
    "CropToAspectRatio": "Crop Image to Aspect Ratio",
    "IfElseValues": "If else with two float values",
}
