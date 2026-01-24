"""
Multi-provider LLM service supporting OpenAI and Anthropic.
"""
import base64
from typing import Any, Dict, List, Literal, Optional, Union

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


class LLMService:
    """
    Multi-provider LLM service for text generation and vision tasks.

    Providers:
    - OpenAI: GPT-4o for vision tasks
    - Anthropic: Claude for reasoning tasks
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        settings = get_settings()

        # OpenAI client
        self.openai_key = openai_api_key or settings.openai_api_key.get_secret_value()
        self.openai_sync = OpenAI(api_key=self.openai_key)
        self.openai_async = AsyncOpenAI(api_key=self.openai_key)

        # Anthropic client (optional)
        anthropic_key = anthropic_api_key or settings.anthropic_api_key.get_secret_value()
        if anthropic_key:
            self.anthropic_sync = Anthropic(api_key=anthropic_key)
            self.anthropic_async = AsyncAnthropic(api_key=anthropic_key)
        else:
            self.anthropic_sync = None
            self.anthropic_async = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Literal["openai", "anthropic"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate text completion using specified provider.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            provider: LLM provider to use
            model: Specific model to use (defaults to provider's best)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        if provider == "anthropic" and self.anthropic_async:
            return await self._complete_anthropic(
                prompt, system_prompt, model, temperature, max_tokens
            )
        return await self._complete_openai(
            prompt, system_prompt, model, temperature, max_tokens
        )

    async def _complete_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate completion using OpenAI."""
        import logging
        logger = logging.getLogger(__name__)

        model = model or "gpt-4o-mini"  # Use mini for lower latency
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(f"OpenAI request: model={model}, prompt_length={len(prompt)}")

        try:
            response = await self.openai_async.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content or ""
            logger.info(f"OpenAI response: length={len(result)}")
            return result
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _complete_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate completion using Anthropic."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.anthropic_async:
            raise ValueError("Anthropic client not configured")

        model = model or "claude-3-5-haiku-20241022"  # Use Haiku for lower latency

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        logger.info(f"Anthropic request: model={model}, prompt_length={len(prompt)}")

        try:
            response = await self.anthropic_async.messages.create(**kwargs)
            result = response.content[0].text if response.content else ""
            logger.info(f"Anthropic response: length={len(result)}")
            return result
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        prompt: str,
        model: str = "gpt-4o-mini",  # Use mini for lower latency
    ) -> str:
        """
        Analyze an image using GPT-4o vision.

        Args:
            image_data: Image as bytes or base64 string
            prompt: Analysis prompt
            model: Model to use (must support vision)

        Returns:
            Analysis result
        """
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode()
        else:
            image_b64 = image_data

        response = await self.openai_async.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    async def extract_product_from_image(self, image_data: Union[bytes, str]) -> dict:
        """
        Extract product information from an image.

        Args:
            image_data: Product image as bytes or base64

        Returns:
            Dictionary with extracted product properties
        """
        prompt = """Analyze this product image and extract the following information:

1. Product name (be specific, include brand if visible)
2. Category (e.g., Electronics, Clothing, Home & Kitchen)
3. Brand (if identifiable)
4. Key features or attributes visible
5. Estimated price range (if any price indicators visible)

Respond in JSON format:
{
    "name": "...",
    "category": "...",
    "brand": "...",
    "features": ["...", "..."],
    "price_indicator": "..." or null
}"""

        response = await self.analyze_image(image_data, prompt)

        # Parse JSON from response
        import json
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "name": response,
            "category": None,
            "brand": None,
            "features": [],
            "price_indicator": None,
        }

    async def extract_product_from_text(self, text: str) -> dict:
        """
        Extract structured product information from text query.

        Args:
            text: User's text query

        Returns:
            Dictionary with extracted product properties
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"extract_product_from_text called with: '{text[:50]}'")

        system_prompt = """You are a product information extractor.
Extract product details from user queries and return structured JSON."""

        prompt = f"""Extract product information from this query:

"{text}"

Return JSON with these fields (use null for unknown):
{{
    "name": "product name",
    "brand": "brand name",
    "category": "product category",
    "price": numeric_price_or_null,
    "currency": "USD/EUR/GBP/etc",
    "gtin": "GTIN/UPC/EAN if mentioned",
    "merchant": "store/seller if mentioned",
    "attributes": {{"key": "value"}}
}}"""

        logger.info("Calling LLM complete()...")
        response = await self.complete(
            prompt,
            system_prompt=system_prompt,
            provider="openai",
            temperature=0.0,
        )
        logger.info(f"LLM response received: {response[:100]}...")

        import json
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "name": text,
            "brand": None,
            "category": None,
            "price": None,
            "currency": None,
            "gtin": None,
            "merchant": None,
            "attributes": {},
        }

    async def rank_search_results(
        self,
        query_properties: dict,
        candidates: List[dict],
        top_k: int = 10,
    ) -> List[dict]:
        """
        Use LLM to rank search candidates by relevance.

        Args:
            query_properties: Extracted properties from user query
            candidates: List of candidate products
            top_k: Number of top results to return

        Returns:
            Ranked list of candidates with confidence scores
        """
        if not candidates:
            return []

        system_prompt = """You are a product matching expert.
Rank products by how well they match the query, considering:
- Name similarity
- Brand match
- Category relevance
- Price compatibility"""

        candidates_text = "\n".join(
            f"{i+1}. {c.get('name', 'Unknown')} - ${c.get('price', 'N/A')} - {c.get('merchant', 'Unknown')}"
            for i, c in enumerate(candidates[:20])  # Limit to 20 for context
        )

        prompt = f"""Query: {query_properties}

Candidates:
{candidates_text}

Return the top {top_k} matches as JSON array with confidence:
[{{"index": 1, "confidence": 0.95}}, ...]

Only include matches with confidence >= 0.5"""

        response = await self.complete(
            prompt,
            system_prompt=system_prompt,
            provider="openai",
            temperature=0.0,
        )

        import json
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                rankings = json.loads(response[start:end])

                ranked_results = []
                for rank in rankings[:top_k]:
                    idx = rank.get("index", 0) - 1
                    if 0 <= idx < len(candidates):
                        result = candidates[idx].copy()
                        result["match_confidence"] = rank.get("confidence", 0.5)
                        ranked_results.append(result)

                return ranked_results
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        # Fallback: return candidates as-is with default confidence
        return [
            {**c, "match_confidence": 0.5}
            for c in candidates[:top_k]
        ]
