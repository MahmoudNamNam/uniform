import io
import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.cluster import KMeans
from skimage import color

app = FastAPI(title="Uniform Segmentation & Static Comparison API")

# --------------------- MODEL SETUP ---------------------
_model: Optional[AutoModelForSemanticSegmentation] = None
_processor: Optional[SegformerImageProcessor] = None


def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon (M1/M2)
        return torch.device("mps")
    return torch.device("cpu")


def load_model_if_needed() -> None:
    """Lazy-loads the SegFormer model and processor."""
    global _model, _processor
    if _model is None or _processor is None:
        model_name = "mattmdjaga/segformer_b2_clothes"
        _processor = SegformerImageProcessor.from_pretrained(model_name)
        _model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        _model.to(get_device())
        _model.eval()


def logits_to_label_ids(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resizes segmentation logits and returns argmax label IDs."""
    upsampled_logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    label_ids = upsampled_logits.argmax(dim=1)
    return label_ids.squeeze(0).to("cpu")


# --------------------- FEATURE EXTRACTION ---------------------
KEEP_CLASSES = [4, 6]  # 4 = upper clothes, 6 = pants


def inference_model(img: Image.Image) -> torch.Tensor:
    """Runs inference and returns class ID mask."""
    load_model_if_needed()
    device = get_device()
    inputs = _processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
    return logits_to_label_ids(logits, img.size[::-1])


def extract_color_features(image: np.ndarray, mask: np.ndarray, keep_classes=KEEP_CLASSES, k=2):
    """Extracts median LAB color + dominant color ratio for shirt/pants."""
    features = {}
    for cls, name in zip(keep_classes, ["upper", "lower"]):
        region_pixels = image[mask == cls]
        if len(region_pixels) == 0:
            features[f"{name}_median"] = None
            features[f"{name}_dominant_ratio"] = 0
            continue

        lab_pixels = color.rgb2lab(region_pixels.reshape(-1, 3).astype(np.float32) / 255.0)
        median_lab = np.median(lab_pixels, axis=0).flatten()
        features[f"{name}_median"] = median_lab

        km = KMeans(n_clusters=k, n_init=3).fit(lab_pixels)
        counts = np.bincount(km.labels_)
        dominant_ratio = counts.max() / counts.sum()
        features[f"{name}_dominant_ratio"] = dominant_ratio
    return features


def deltaE(lab1, lab2):
    """Computes perceptual color distance (CIEDE2000)."""
    if lab1 is None or lab2 is None:
        return np.inf
    return color.deltaE_ciede2000(lab1.reshape(1, 3), lab2.reshape(1, 3))[0]


def compare_features(featA, featB):
    """Computes similarity score between two color feature sets."""
    top_diff = deltaE(featA["upper_median"], featB["upper_median"])
    bottom_diff = deltaE(featA["lower_median"], featB["lower_median"])
    top_bottom_diff = deltaE(featA["upper_median"], featB["lower_median"])
    bottom_top_diff = deltaE(featA["lower_median"], featB["upper_median"])
    same_dress_diff = deltaE(featA["upper_median"], featA["lower_median"])

    final_score = (
        10 * top_diff
        + 10 * bottom_diff
        + 5 * top_bottom_diff
        + 5 * bottom_top_diff
        + 5 * same_dress_diff
    ) / 35

    similarity = max(0, min(1.0, 1.5 - final_score / 10))
    return similarity


# --------------------- ENDPOINTS ---------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/compare")
async def compare_static_reference(
    test_image: UploadFile = File(..., description="Image to compare against static reference"),
):
    """
    Compare uploaded uniform image against a static reference (ref.jpg) in project root.
    """
    try:
        load_model_if_needed()

        # Load static reference
        ref_path = "./ref-2.jpg"
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=404, detail=f"Reference image not found at {ref_path}")

        ref_img = Image.open(ref_path).convert("RGB")
        test_img = Image.open(io.BytesIO(await test_image.read())).convert("RGB")

        # Run segmentation
        ref_mask = inference_model(ref_img)
        test_mask = inference_model(test_img)

        # Extract color features
        ref_feat = extract_color_features(np.array(ref_img), ref_mask.numpy())
        test_feat = extract_color_features(np.array(test_img), test_mask.numpy())

        # Compare
        similarity = compare_features(ref_feat, test_feat)
        match = similarity > 0.6

        return JSONResponse({
            "similarity_score": round(float(similarity), 3),
            "match": bool(match),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

