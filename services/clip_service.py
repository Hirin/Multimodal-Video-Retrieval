"""
CLIP service for text encoding and similarity search.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from utils.logger import get_logger
from utils.helpers import l2_normalize_rows

logger = get_logger('clip_service')

# CLIP runtime state (singleton pattern)
_CLIP_RT: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "backend": None
}


def ensure_clip_text_encoder(model_name: str = 'ViT-B-32', pretrained: str = 'openai') -> None:
    """
    Ensure CLIP text encoder is loaded (singleton pattern).

    Args:
        model_name: Name of the CLIP model
        pretrained: Pretrained weights to use

    Raises:
        RuntimeError: If CLIP text encoder cannot be loaded
    """
    if _CLIP_RT["model"] is not None:
        return

    # Try open_clip first
    try:
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.to(device)
        model.eval()
        _CLIP_RT.update({
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "backend": "open_clip"
        })
        logger.info(f"Loaded CLIP model (open_clip) on {device}")
        return
    except Exception as e:
        logger.warning(f"Failed to load open_clip: {e}")

    # Fallback to clip
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(model_name, device=device, jit=False)
        _CLIP_RT.update({
            "model": model,
            "tokenizer": clip.tokenize,
            "device": device,
            "backend": "clip"
        })
        logger.info(f"Loaded CLIP model (clip) on {device}")
        return
    except Exception as e:
        logger.error(f"Failed to load clip: {e}")
        raise RuntimeError("CLIP text encoder (open_clip/clip) not found.")


def encode_text(texts: List[str]) -> np.ndarray:
    """
    Encode texts using CLIP text encoder.

    Args:
        texts: List of text strings to encode

    Returns:
        Normalized text embeddings as numpy array

    Raises:
        RuntimeError: If CLIP encoder is not available
    """
    ensure_clip_text_encoder()
    import torch

    try:
        toks = _CLIP_RT["tokenizer"](texts)
        with torch.no_grad():
            toks = toks.to(_CLIP_RT["device"])
            feats = _CLIP_RT["model"].encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        embeddings = feats.detach().cpu().numpy().astype("float32")
        logger.debug(f"Encoded {len(texts)} texts, shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        raise


def compute_similarity(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute cosine similarity between text and image embeddings.

    Args:
        text_embeddings: Text embeddings (N, D)
        image_embeddings: Image embeddings (M, D)
        normalize: Whether to L2 normalize before computing similarity

    Returns:
        Similarity matrix (N, M)
    """
    if normalize:
        text_embeddings = l2_normalize_rows(text_embeddings)
        image_embeddings = l2_normalize_rows(image_embeddings)

    similarity = np.dot(text_embeddings, image_embeddings.T)
    return similarity


def reset_clip_model() -> None:
    """
    Reset CLIP model to free memory.
    """
    global _CLIP_RT
    if _CLIP_RT["model"] is not None:
        import torch
        del _CLIP_RT["model"]
        del _CLIP_RT["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _CLIP_RT = {"model": None, "tokenizer": None, "device": None, "backend": None}
        logger.info("CLIP model reset and memory freed")


def get_clip_info() -> Dict[str, Any]:
    """
    Get information about the loaded CLIP model.

    Returns:
        Dictionary with model information
    """
    return {
        "loaded": _CLIP_RT["model"] is not None,
        "device": _CLIP_RT.get("device"),
        "backend": _CLIP_RT.get("backend")
    }
