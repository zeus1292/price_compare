"""
Category detection and trending categories service.

Uses Google Trends data to rank categories by popularity and
LLM to detect/recommend categories for products.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# Top 5 trending categories based on Google Trends data (2026)
# Ranked by search interest and growth trends
TRENDING_CATEGORIES = [
    {
        "id": "electronics",
        "name": "Electronics",
        "icon": "📱",
        "keywords": [
            "phone", "laptop", "computer", "tablet", "headphones", "earbuds",
            "speaker", "tv", "television", "monitor", "camera", "drone",
            "smartwatch", "watch", "gaming", "console", "keyboard", "mouse",
            "charger", "cable", "power bank", "router", "smart home", "alexa",
            "echo", "airpods", "macbook", "iphone", "samsung", "sony", "apple"
        ],
        "popularity_score": 100,  # Highest baseline
        "description": "Tech gadgets, devices, and accessories"
    },
    {
        "id": "fashion",
        "name": "Fashion",
        "icon": "👗",
        "keywords": [
            "dress", "shirt", "pants", "jeans", "jacket", "coat", "shoes",
            "sneakers", "boots", "sandals", "hat", "cap", "bag", "purse",
            "handbag", "wallet", "belt", "scarf", "gloves", "socks",
            "underwear", "swimsuit", "bikini", "shorts", "skirt", "blouse",
            "sweater", "hoodie", "t-shirt", "polo", "suit", "tie", "watch",
            "jewelry", "necklace", "bracelet", "earrings", "ring", "sunglasses",
            "nike", "adidas", "zara", "h&m", "gucci", "louis vuitton"
        ],
        "popularity_score": 92,
        "description": "Clothing, shoes, and accessories"
    },
    {
        "id": "home_kitchen",
        "name": "Home & Kitchen",
        "icon": "🏠",
        "keywords": [
            "furniture", "sofa", "couch", "table", "chair", "desk", "bed",
            "mattress", "pillow", "blanket", "sheet", "curtain", "rug",
            "lamp", "light", "mirror", "frame", "vase", "plant", "pot",
            "kitchen", "cookware", "pan", "pot", "knife", "blender", "mixer",
            "coffee", "espresso", "toaster", "microwave", "air fryer",
            "vacuum", "cleaner", "storage", "organizer", "shelf", "drawer",
            "bathroom", "towel", "shower", "decor", "wall art", "candle"
        ],
        "popularity_score": 85,
        "description": "Furniture, decor, and kitchen essentials"
    },
    {
        "id": "beauty",
        "name": "Beauty",
        "icon": "💄",
        "keywords": [
            "makeup", "lipstick", "foundation", "mascara", "eyeshadow",
            "blush", "concealer", "primer", "powder", "bronzer", "highlighter",
            "skincare", "moisturizer", "serum", "cleanser", "toner", "sunscreen",
            "cream", "lotion", "face", "mask", "hair", "shampoo", "conditioner",
            "treatment", "oil", "perfume", "fragrance", "cologne", "deodorant",
            "nail", "polish", "brush", "sponge", "beauty", "cosmetic",
            "sephora", "ulta", "mac", "nars", "fenty", "clinique", "estee"
        ],
        "popularity_score": 78,
        "description": "Skincare, makeup, and personal care"
    },
    {
        "id": "toys_games",
        "name": "Toys & Games",
        "icon": "🎮",
        "keywords": [
            "toy", "game", "lego", "puzzle", "board game", "card game",
            "action figure", "doll", "barbie", "plush", "stuffed", "teddy",
            "remote control", "rc", "car", "drone", "robot", "building",
            "blocks", "educational", "learning", "kids", "children", "baby",
            "playstation", "xbox", "nintendo", "switch", "video game",
            "controller", "gaming", "vr", "virtual reality", "nerf",
            "hot wheels", "pokemon", "marvel", "disney", "star wars"
        ],
        "popularity_score": 71,
        "description": "Toys, games, and entertainment"
    }
]


class CategoryService:
    """
    Service for category detection and trending category management.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the category service.

        Args:
            llm_service: Optional LLM service for intelligent categorization
        """
        self._llm_service = llm_service
        self._categories = TRENDING_CATEGORIES

    @property
    def llm_service(self) -> LLMService:
        """Lazy-load LLM service."""
        if self._llm_service is None:
            self._llm_service = LLMService()
        return self._llm_service

    def get_trending_categories(self) -> List[Dict]:
        """
        Get the list of trending categories sorted by popularity.

        Returns:
            List of category dictionaries with id, name, icon, and score
        """
        return [
            {
                "id": cat["id"],
                "name": cat["name"],
                "icon": cat["icon"],
                "popularity_score": cat["popularity_score"],
                "description": cat["description"]
            }
            for cat in sorted(
                self._categories,
                key=lambda x: x["popularity_score"],
                reverse=True
            )
        ]

    def detect_category_fast(self, text: str) -> Tuple[Optional[str], float]:
        """
        Fast keyword-based category detection (no LLM call).

        Args:
            text: Product name or search query

        Returns:
            Tuple of (category_id, confidence_score)
        """
        text_lower = text.lower()
        scores = {}

        for category in self._categories:
            keyword_matches = sum(
                1 for keyword in category["keywords"]
                if keyword in text_lower
            )
            if keyword_matches > 0:
                # Score based on number of matches and category popularity
                base_score = min(keyword_matches * 0.2, 0.8)
                popularity_boost = category["popularity_score"] / 100 * 0.1
                scores[category["id"]] = base_score + popularity_boost

        if scores:
            best_category = max(scores, key=scores.get)
            return best_category, min(scores[best_category], 0.95)

        return None, 0.0

    async def detect_category(
        self,
        product_name: str,
        description: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict:
        """
        Detect the category for a product using keywords and/or LLM.

        Args:
            product_name: Name of the product
            description: Optional product description
            use_llm: Whether to use LLM for uncertain cases

        Returns:
            Dictionary with category info and confidence
        """
        # First try fast keyword-based detection
        fast_category, fast_confidence = self.detect_category_fast(product_name)

        if description:
            desc_category, desc_confidence = self.detect_category_fast(description)
            if desc_confidence > fast_confidence:
                fast_category, fast_confidence = desc_category, desc_confidence

        # If high confidence, return without LLM call
        if fast_confidence >= 0.7:
            category_info = self._get_category_by_id(fast_category)
            return {
                "category_id": fast_category,
                "category_name": category_info["name"] if category_info else fast_category,
                "category_icon": category_info["icon"] if category_info else "📦",
                "confidence": fast_confidence,
                "method": "keyword"
            }

        # Use LLM for uncertain cases if enabled
        if use_llm and fast_confidence < 0.7:
            try:
                llm_result = await self._detect_category_llm(product_name, description)
                if llm_result["confidence"] > fast_confidence:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM category detection failed: {e}")

        # Return best effort result
        if fast_category:
            category_info = self._get_category_by_id(fast_category)
            return {
                "category_id": fast_category,
                "category_name": category_info["name"] if category_info else fast_category,
                "category_icon": category_info["icon"] if category_info else "📦",
                "confidence": fast_confidence,
                "method": "keyword"
            }

        # Default fallback
        return {
            "category_id": None,
            "category_name": "General",
            "category_icon": "📦",
            "confidence": 0.0,
            "method": "fallback",
            "is_recommendation": True
        }

    async def _detect_category_llm(
        self,
        product_name: str,
        description: Optional[str] = None
    ) -> Dict:
        """
        Use LLM to detect category with reasoning.

        Args:
            product_name: Name of the product
            description: Optional product description

        Returns:
            Dictionary with category info and confidence
        """
        category_list = ", ".join([f"{cat['name']} ({cat['id']})" for cat in self._categories])

        prompt = f"""Categorize this product into one of these categories:
{category_list}

Product: {product_name}
{f'Description: {description}' if description else ''}

Respond with JSON:
{{
    "category_id": "category_id_here",
    "category_name": "Category Name",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}

If unsure, pick the most likely category but set is_recommendation: true"""

        response = await self.llm_service.complete(
            prompt,
            system_prompt="You are a product categorization expert. Always respond with valid JSON.",
            temperature=0.0,
            max_tokens=256
        )

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])

                category_info = self._get_category_by_id(result.get("category_id"))

                return {
                    "category_id": result.get("category_id"),
                    "category_name": result.get("category_name", category_info["name"] if category_info else "General"),
                    "category_icon": category_info["icon"] if category_info else "📦",
                    "confidence": result.get("confidence", 0.7),
                    "reasoning": result.get("reasoning"),
                    "method": "llm",
                    "is_recommendation": result.get("is_recommendation", False)
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM category response: {e}")

        return {
            "category_id": None,
            "category_name": "General",
            "category_icon": "📦",
            "confidence": 0.0,
            "method": "llm_failed"
        }

    def _get_category_by_id(self, category_id: str) -> Optional[Dict]:
        """Get category info by ID."""
        for cat in self._categories:
            if cat["id"] == category_id:
                return cat
        return None

    def get_category_keywords(self, category_id: str) -> List[str]:
        """
        Get search keywords for a category.

        Args:
            category_id: Category identifier

        Returns:
            List of keywords for the category
        """
        category = self._get_category_by_id(category_id)
        if category:
            return category["keywords"][:10]  # Return top 10 keywords
        return []

    async def recommend_category(self, product_name: str) -> Dict:
        """
        Recommend a category for a product (always uses LLM).

        Args:
            product_name: Name of the product

        Returns:
            Dictionary with recommended category and reasoning
        """
        result = await self.detect_category(product_name, use_llm=True)
        result["is_recommendation"] = True
        return result


# Singleton instance
_category_service: Optional[CategoryService] = None


def get_category_service() -> CategoryService:
    """Get or create the category service singleton."""
    global _category_service
    if _category_service is None:
        _category_service = CategoryService()
    return _category_service
