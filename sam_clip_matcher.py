import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip

class SAMCLIPMatcher:
    def __init__(self, sam_checkpoint, model_type="vit_b", clip_model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device, download_root="checkpoints")

    def process(self, image: Image.Image, prompt: str):
        np_image = np.array(image.convert("RGB"))
        masks = self.mask_generator.generate(np_image)

        patches = []
        valid_indices = []

        for idx, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            x, y, w, h = map(int, mask_data["bbox"])
            masked_patch = np_image.copy()
            masked_patch[~mask] = 127
            patch = masked_patch[y:y+h, x:x+w]

            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch_pil = Image.fromarray(patch)
                patches.append(self.preprocess(patch_pil).unsqueeze(0).to(self.device))
                valid_indices.append(idx)

        if not patches:
            return None, None, "No valid mask patches."

        image_features = torch.cat(patches)
        text = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_features)
            text_features = self.clip_model.encode_text(text)

        similarity = image_features @ text_features.T
        best_idx_in_valid = similarity.argmax().item()
        best_mask_idx = valid_indices[best_idx_in_valid]
        best_mask = masks[best_mask_idx]["segmentation"]

        overlay = np.zeros_like(np_image)
        overlay[best_mask] = [0, 255, 0]
        vis_image = cv2.addWeighted(np_image, 0.7, overlay, 0.3, 0)
        return vis_image, best_mask, None
