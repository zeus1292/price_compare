"""
Property Extraction Agent for extracting product properties from queries.

Supports:
- Text queries: LLM-based extraction
- URL input: Fetch and parse webpage
- Image input: CLIP embedding for fast visual search (falls back to GPT-4o)
"""
import base64
import logging
from typing import Any, List, Optional

import httpx
from langsmith import traceable

from src.agents.base_agent import BaseAgent
from src.services.llm_service import LLMService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropertyExtractorAgent(BaseAgent):
    """
    Agent for extracting product properties from user input.

    Supports:
    - Text queries: Extract product name, brand, price, etc.
    - URL input: Fetch webpage and extract product metadata
    - Image input: Use GPT-4o vision to identify product

    Output:
        {
            "name": "Product Name",
            "brand": "Brand Name",
            "category": "Electronics",
            "price": 29.99,
            "currency": "USD",
            "gtin": "0123456789012",
            "merchant": "Amazon",
            "attributes": {...}
        }
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        super().__init__(name="PropertyExtractor", llm_service=llm_service)

    @traceable(name="property_extraction", run_type="chain")
    async def _execute_impl(self, state: dict) -> dict:
        """
        Extract product properties based on input type.

        Args:
            state: Must contain 'user_input' and 'input_type'
                   Optional 'use_clip' for image search (default True)

        Returns:
            State with 'extracted_properties' and 'extraction_confidence'
            For images with CLIP: also includes 'image_embedding' for direct search
        """
        self.validate_state(state, ["user_input", "input_type"])

        user_input = state["user_input"]
        input_type = state["input_type"]
        use_clip = state.get("use_clip", True)

        logger.info(f"Extracting properties from {input_type} input (use_clip={use_clip})")

        try:
            if input_type == "text":
                properties = await self._extract_from_text(user_input)
                image_embedding = None
            elif input_type == "url":
                properties = await self._extract_from_url(user_input)
                image_embedding = None
            elif input_type == "image":
                # Try CLIP first for fast image embedding
                if use_clip:
                    properties, image_embedding = await self._extract_from_image_clip(user_input)
                else:
                    properties = await self._extract_from_image(user_input)
                    image_embedding = None
            else:
                raise ValueError(f"Unknown input type: {input_type}")

            # Normalize extracted properties
            normalized = self._normalize_properties(properties)

            # Fall back to raw query as name when LLM returns null for text input
            if not normalized.get("name") and input_type == "text":
                normalized["name"] = user_input

            # Calculate extraction confidence
            confidence = self._calculate_confidence(normalized)

            result = {
                "extracted_properties": normalized,
                "extraction_confidence": confidence,
            }

            # Include image embedding if available (for direct CLIP search)
            if image_embedding is not None:
                result["image_embedding"] = image_embedding

            return self.update_state(state, result)

        except Exception as e:
            logger.error(f"Property extraction failed: {e}")
            return self.update_state(state, {
                "extracted_properties": {"name": user_input if input_type == "text" else "Unknown Product"},
                "extraction_confidence": 0.3,
                "extraction_error": str(e),
            })

    async def _extract_from_text(self, text: str) -> dict:
        """
        Extract product properties from text query.

        Uses LLM to parse natural language queries.
        """
        logger.info(f"Extracting properties from text: '{text[:100]}'")
        try:
            logger.info("Calling LLM for text extraction...")
            result = await self.llm.extract_product_from_text(text)
            logger.info(f"LLM extraction successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Text extraction LLM call failed: {type(e).__name__}: {e}", exc_info=True)
            # Re-raise to let the caller know the LLM failed
            raise

    async def _extract_from_url(self, url: str) -> dict:
        """
        Extract product properties from URL.

        Fetches webpage and extracts product metadata.
        """
        try:
            # Fetch webpage content
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                html_content = response.text

            # Extract using LLM
            system_prompt = """You are a product information extractor.
Extract product details from HTML content and return structured JSON."""

            prompt = f"""Extract product information from this webpage HTML:

URL: {url}
HTML (first 5000 chars):
{html_content[:5000]}

Return JSON with:
{{
    "name": "product name",
    "brand": "brand name or null",
    "category": "product category",
    "price": numeric_price_or_null,
    "currency": "USD/EUR/GBP/etc",
    "merchant": "store name",
    "image_url": "main product image URL or null",
    "attributes": {{"key": "value"}}
}}"""

            response = await self.call_llm(
                prompt,
                system_prompt=system_prompt,
                provider="openai",
            )

            # Parse JSON from response
            import json
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    properties = json.loads(response[start:end])
                    properties["source_url"] = url
                    return properties
            except json.JSONDecodeError:
                pass

            return {"name": url, "source_url": url}

        except Exception as e:
            logger.error(f"URL extraction failed: {e}")
            return {"name": url, "source_url": url, "error": str(e)}

    async def _extract_from_image_clip(
        self,
        image_data: str,
    ) -> tuple:
        """
        Extract image embedding using CLIP for fast visual search.

        Returns:
            Tuple of (properties dict, image embedding list)
        """
        try:
            from src.services.clip_service import get_clip_service

            clip_service = get_clip_service()

            if not clip_service.is_available():
                logger.warning("CLIP not available, falling back to GPT-4o")
                properties = await self._extract_from_image(image_data)
                return properties, None

            logger.info("Using CLIP for fast image embedding")

            # Get image bytes
            image_bytes = await self._get_image_bytes(image_data)

            # Generate CLIP embedding
            embedding = clip_service.embed_image(image_bytes)

            # For CLIP-based search, we don't extract properties from the image
            # The embedding is used directly for similarity search
            # We'll return minimal properties and let the search fill in details
            properties = {
                "name": "Image Search Query",
                "search_type": "visual",
            }

            logger.info(f"CLIP embedding generated: {len(embedding)} dimensions")
            return properties, embedding

        except ImportError:
            logger.warning("CLIP/sentence-transformers not installed, using GPT-4o")
            properties = await self._extract_from_image(image_data)
            return properties, None
        except Exception as e:
            logger.warning(f"CLIP embedding failed: {e}, falling back to GPT-4o")
            properties = await self._extract_from_image(image_data)
            return properties, None

    async def _get_image_bytes(self, image_data: str) -> bytes:
        """Convert various image input formats to bytes."""
        if image_data.startswith("http"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_data)
                response.raise_for_status()
                return response.content
        elif image_data.startswith("data:"):
            _, data = image_data.split(",", 1)
            return base64.b64decode(data)
        else:
            try:
                return base64.b64decode(image_data)
            except Exception:
                with open(image_data, "rb") as f:
                    return f.read()

    async def _extract_from_image(self, image_data: str) -> dict:
        """
        Extract product properties from image using GPT-4o vision.

        Fallback when CLIP is not available.
        """
        # Handle different image input formats
        if image_data.startswith("http"):
            # URL to image - fetch it first
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_data)
                response.raise_for_status()
                image_bytes = response.content
        elif image_data.startswith("data:"):
            # Data URL - extract base64
            _, data = image_data.split(",", 1)
            image_bytes = base64.b64decode(data)
        else:
            # Assume base64 encoded
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                # Treat as file path
                with open(image_data, "rb") as f:
                    image_bytes = f.read()

        # Use vision model to extract properties
        return await self.llm.extract_product_from_image(image_bytes)

    def _normalize_properties(self, properties: dict) -> dict:
        """
        Normalize extracted properties.

        - Ensure consistent field names
        - Clean and validate values
        - Handle missing fields
        """
        normalized = {}

        # Name (required)
        name = properties.get("name") or properties.get("product_name", "")
        normalized["name"] = name.strip() if name else ""

        # Brand
        brand = properties.get("brand") or properties.get("manufacturer")
        if brand:
            normalized["brand"] = brand.strip()

        # Category
        category = properties.get("category") or properties.get("product_type")
        if category:
            normalized["category"] = category.strip()

        # Price
        price = properties.get("price")
        if price is not None:
            try:
                if isinstance(price, str):
                    # Clean price string
                    price = price.replace("$", "").replace(",", "").strip()
                normalized["price"] = float(price)
            except (ValueError, TypeError):
                pass

        # Currency
        currency = properties.get("currency", "USD")
        normalized["currency"] = currency.upper() if currency else "USD"

        # GTIN/UPC/EAN
        gtin = (
            properties.get("gtin") or
            properties.get("upc") or
            properties.get("ean") or
            properties.get("barcode")
        )
        if gtin:
            normalized["gtin"] = str(gtin).strip()

        # Merchant
        merchant = properties.get("merchant") or properties.get("seller")
        if merchant:
            normalized["merchant"] = merchant.strip()

        # Source URL
        if properties.get("source_url"):
            normalized["source_url"] = properties["source_url"]

        # Image URL
        if properties.get("image_url"):
            normalized["image_url"] = properties["image_url"]

        # Additional attributes
        attributes = properties.get("attributes", {})
        if attributes and isinstance(attributes, dict):
            normalized["attributes"] = attributes

        # Features (from image extraction)
        features = properties.get("features", [])
        if features:
            normalized["features"] = features

        return normalized

    def _calculate_confidence(self, properties: dict) -> float:
        """
        Calculate confidence score for extracted properties.

        Higher confidence when more fields are extracted.
        """
        weights = {
            "name": 0.3,
            "brand": 0.15,
            "category": 0.1,
            "price": 0.15,
            "gtin": 0.2,
            "merchant": 0.1,
        }

        score = 0.0
        for field, weight in weights.items():
            if properties.get(field):
                score += weight

        # Bonus for name quality
        name = properties.get("name", "")
        if len(name) > 10:
            score += 0.1
        if len(name) > 30:
            score += 0.05

        return min(1.0, score)

    @traceable(name="extract_text_properties")
    async def extract_text_properties(self, text: str) -> dict:
        """
        Standalone method for text property extraction.

        Can be used as a tool by other agents.
        """
        properties = await self._extract_from_text(text)
        return self._normalize_properties(properties)

    @traceable(name="extract_image_properties")
    async def extract_image_properties(self, image_data: str) -> dict:
        """
        Standalone method for image property extraction.

        Can be used as a tool by other agents.
        """
        properties = await self._extract_from_image(image_data)
        return self._normalize_properties(properties)
