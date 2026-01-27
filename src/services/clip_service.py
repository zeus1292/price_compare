"""
CLIP Embedding Service for image-based product search.

Uses CLIP (Contrastive Language-Image Pre-training) to generate
embeddings that work for both images and text, enabling:
- Image → similar products search
- Text → image matching

Much faster than GPT-4o vision for image search.
"""
import base64
import io
import logging
from typing import List, Optional, Union

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy load the model to avoid startup overhead
_clip_model = None
_clip_processor = None


def get_clip_model():
    """
    Lazy load CLIP model on first use.
    Uses sentence-transformers which handles CLIP efficiently.
    """
    global _clip_model

    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading CLIP model (clip-ViT-B-32)...")
            _clip_model = SentenceTransformer("clip-ViT-B-32")
            logger.info("CLIP model loaded successfully")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    return _clip_model


class CLIPService:
    """
    Service for generating CLIP embeddings from images and text.

    CLIP embeddings enable:
    - Direct image-to-image similarity search
    - Cross-modal search (image query, text results and vice versa)
    - Much faster than LLM-based image analysis
    """

    def __init__(self):
        self._model = None
        self._embedding_dim = 512  # CLIP ViT-B-32 output dimension

    @property
    def model(self):
        """Lazy load model on first access."""
        if self._model is None:
            self._model = get_clip_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def is_available(self) -> bool:
        """Check if CLIP is available."""
        try:
            _ = self.model
            return True
        except Exception:
            return False

    def embed_image(self, image_data: Union[bytes, str, Image.Image]) -> List[float]:
        """
        Generate CLIP embedding from an image.

        Args:
            image_data: Image as bytes, base64 string, file path, or PIL Image

        Returns:
            List of floats (512-dimensional embedding)
        """
        # Convert to PIL Image
        pil_image = self._to_pil_image(image_data)

        # Generate embedding
        embedding = self.model.encode(pil_image, convert_to_numpy=True)

        return embedding.tolist()

    def embed_images_batch(
        self,
        images: List[Union[bytes, str, Image.Image]],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple images.

        Args:
            images: List of images (bytes, base64, paths, or PIL Images)
            batch_size: Number of images to process at once

        Returns:
            List of embeddings
        """
        pil_images = [self._to_pil_image(img) for img in images]

        embeddings = self.model.encode(
            pil_images,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(pil_images) > 10,
        )

        return [emb.tolist() for emb in embeddings]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate CLIP embedding from text.

        This allows cross-modal search: text query against image embeddings.

        Args:
            text: Text to embed

        Returns:
            List of floats (512-dimensional embedding)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple texts.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )

        return [emb.tolist() for emb in embeddings]

    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score between 0 and 1
        """
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _to_pil_image(self, image_data: Union[bytes, str, Image.Image]) -> Image.Image:
        """
        Convert various image formats to PIL Image.

        Handles:
        - PIL Image (passthrough)
        - Bytes (raw image data)
        - Base64 string
        - Data URL (data:image/...;base64,...)
        - File path
        """
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")

        if isinstance(image_data, str):
            # Check if it's a data URL
            if image_data.startswith("data:"):
                # Extract base64 part
                _, data = image_data.split(",", 1)
                image_bytes = base64.b64decode(data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Check if it's base64
            try:
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                pass

            # Assume it's a file path
            return Image.open(image_data).convert("RGB")

        raise ValueError(f"Unsupported image format: {type(image_data)}")


# Singleton instance
_clip_service: Optional[CLIPService] = None


def get_clip_service() -> CLIPService:
    """Get or create the CLIP service singleton."""
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPService()
    return _clip_service
