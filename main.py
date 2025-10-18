import io
import os
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.cluster import KMeans
from skimage import color

# ---------------------------------------------------------
# APP INITIALIZATION + CORS
# ---------------------------------------------------------
app = FastAPI(title="Uniform Segmentation & Static Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# MODEL SETUP
# ---------------------------------------------------------
_model: Optional[AutoModelForSemanticSegmentation] = None
_processor: Optional[SegformerImageProcessor] = None

def get_device() -> torch.device:
    return torch.device("cpu")

def load_model_if_needed() -> None:
    """Lazy-load SegFormer model."""
    global _model, _processor
    if _model is None or _processor is None:
        model_name = "mattmdjaga/segformer_b2_clothes"
        _processor = SegformerImageProcessor.from_pretrained(model_name)
        _model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        _model.to(get_device())
        _model.eval()

def logits_to_label_ids(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resizes logits and returns argmax label IDs."""
    upsampled = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    label_ids = upsampled.argmax(dim=1)
    return label_ids.squeeze(0).to("cpu")

# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
KEEP_CLASSES = [4, 5, 6, 7, 8]  # dress, coat, shirt, pants, skirt

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
    """Extracts LAB median colors for visible clothing regions."""
    features = {}
    found = 0
    for cls in keep_classes:
        region = image[mask == cls]
        if len(region) == 0:
            continue
        found += 1
        lab_pixels = color.rgb2lab(region.reshape(-1, 3).astype(np.float32) / 255.0)
        median_lab = np.median(lab_pixels, axis=0).flatten()
        features[f"class_{cls}_median"] = median_lab
        # Dominant color ratio
        km = KMeans(n_clusters=k, n_init=3).fit(lab_pixels)
        counts = np.bincount(km.labels_)
        dominant_ratio = counts.max() / counts.sum()
        features[f"class_{cls}_dominant_ratio"] = dominant_ratio
    if found == 0:
        return None
    return features

def deltaE(lab1, lab2):
    """Computes perceptual color distance (CIEDE2000)."""
    if lab1 is None or lab2 is None:
        return np.inf
    return color.deltaE_ciede2000(lab1.reshape(1, 3), lab2.reshape(1, 3))[0]

def compare_features(featA, featB):
    """Computes similarity score between two color feature sets."""
    if featA is None or featB is None:
        return 0.0

    common = set(featA.keys()) & set(featB.keys())
    common = [k for k in common if k.endswith("_median")]
    if not common:
        return 0.0

    diffs = [deltaE(featA[k], featB[k]) for k in common]
    mean_diff = np.mean(diffs)
    similarity = max(0.0, min(1.0, 1.5 - mean_diff / 10))
    return similarity

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/compare")
async def compare_static_reference(
    test_image: UploadFile = File(..., description="Image to compare against static reference"),
):
    """
    Compare uploaded uniform image against static reference (ref-2.jpg).
    """
    try:
        load_model_if_needed()

        # Static reference path
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
