"""
Live Search Agent using Tavily for web search.

Optimized for speed with:
- Parallel LLM calls for result parsing
- Basic search depth option for faster results
- Heuristic-based extraction as fallback
"""
import asyncio
import logging
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

from langsmith import traceable
from tavily import TavilyClient

from src.agents.base_agent import BaseAgent
from src.config.settings import get_settings
from src.services.llm_service import LLMService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveSearcherAgent(BaseAgent):
    """
    Agent for searching the web when database has no confident match.

    Uses Tavily API for web search and LLM for result parsing.

    Output:
        {
            "live_search_results": [...],
            "live_search_triggered": True,
            "search_time_ms": 1234
        }
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        tavily_api_key: Optional[str] = None,
    ):
        super().__init__(name="LiveSearcher", llm_service=llm_service)

        settings = get_settings()
        self.tavily_key = tavily_api_key or settings.tavily_api_key.get_secret_value()

        if self.tavily_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_key)
        else:
            self.tavily_client = None
            logger.warning("Tavily API key not configured - live search disabled")

    @traceable(name="live_search", run_type="chain")
    async def _execute_impl(self, state: dict) -> dict:
        """
        Execute live web search.

        Args:
            state: Must contain 'extracted_properties'
                   Optional 'fast_mode' (default True) - use heuristics instead of LLM

        Returns:
            State with 'live_search_results' and metadata
        """
        self.validate_state(state, ["extracted_properties"])

        if not self.tavily_client:
            return self.update_state(state, {
                "live_search_results": [],
                "live_search_triggered": False,
                "live_search_error": "Tavily API not configured",
            })

        properties = state["extracted_properties"]
        fast_mode = state.get("fast_mode", True)  # Default to fast mode
        start_time = time.perf_counter()

        try:
            # Build search query
            search_query = self._build_search_query(properties)
            logger.info(f"Live search query: {search_query} (fast_mode={fast_mode})")

            # Execute Tavily search (basic depth for speed)
            tavily_results = await self._tavily_search(
                search_query,
                max_results=8,  # Fewer results for speed
                search_depth="basic" if fast_mode else "advanced",
            )

            # Parse results - fast mode uses heuristics, slow mode uses LLM
            if fast_mode:
                parsed_products = self._parse_results_fast(tavily_results, properties)
            else:
                parsed_products = await self._parse_results_with_llm(
                    tavily_results, properties
                )

            # Verify matches
            verified = self._verify_matches(parsed_products, properties)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"Live search completed in {elapsed_ms}ms (fast_mode={fast_mode})")

            return self.update_state(state, {
                "live_search_results": verified,
                "live_search_triggered": True,
                "live_search_query": search_query,
                "live_search_raw_count": len(tavily_results),
                "search_time_ms": elapsed_ms,
            })

        except Exception as e:
            logger.error(f"Live search failed: {e}")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            return self.update_state(state, {
                "live_search_results": [],
                "live_search_triggered": True,
                "live_search_error": str(e),
                "search_time_ms": elapsed_ms,
            })

    def _build_search_query(self, properties: dict) -> str:
        """
        Build optimal search query from properties.
        """
        parts = []

        # Product name is primary
        if properties.get("name"):
            parts.append(properties["name"])

        # Add brand if not in name
        brand = properties.get("brand", "")
        name = properties.get("name", "")
        if brand and brand.lower() not in name.lower():
            parts.insert(0, brand)

        # Add category for context
        if properties.get("category"):
            parts.append(properties["category"])

        # Add "buy" or "price" to get shopping results
        parts.append("buy price")

        return " ".join(parts)

    @traceable(name="tavily_search")
    async def _tavily_search(
        self,
        query: str,
        max_results: int = 8,
        search_depth: str = "basic",
    ) -> List[dict]:
        """
        Execute Tavily search.

        Args:
            query: Search query
            max_results: Maximum results to return
            search_depth: "basic" (fast) or "advanced" (thorough)
        """
        if not self.tavily_client:
            return []

        try:
            response = self.tavily_client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_images=True,
                include_domains=[
                    "amazon.com",
                    "ebay.com",
                    "walmart.com",
                    "target.com",
                    "bestbuy.com",
                    "newegg.com",
                ],
            )

            results = response.get("results", [])
            images = response.get("images", [])

            # Attach images to results if available
            for i, result in enumerate(results):
                if i < len(images):
                    result["image_url"] = images[i]

            return results

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def _parse_results_fast(
        self,
        results: List[dict],
        query_properties: dict,
    ) -> List[dict]:
        """
        Fast parsing using heuristics instead of LLM.
        Extracts product info from title, URL, and content using regex.
        """
        parsed = []
        query_name = query_properties.get("name", "").lower()

        for result in results:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            image_url = result.get("image_url")

            if not title:
                continue

            # Extract price using regex
            price = self._extract_price_fast(title + " " + content)

            # Extract merchant from URL
            merchant = self._extract_merchant_from_url(url)

            # Calculate match confidence based on title similarity
            confidence = self._calculate_similarity(query_name, title.lower())

            parsed.append({
                "name": title,
                "price": price,
                "currency": "USD",
                "merchant": merchant,
                "source_url": url,
                "image_url": image_url,
                "match_source": "live",
                "match_confidence": confidence,
            })

        return parsed

    def _extract_price_fast(self, text: str) -> Optional[float]:
        """
        Extract price from text using regex.
        """
        # Match patterns like $99.99, $1,234.56, USD 99.99
        patterns = [
            r'\$([0-9,]+\.?\d*)',
            r'USD\s*([0-9,]+\.?\d*)',
            r'Price:\s*\$?([0-9,]+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(',', '')
                    return float(price_str)
                except ValueError:
                    continue

        return None

    def _calculate_similarity(self, query: str, title: str) -> float:
        """
        Calculate simple similarity score between query and title.
        """
        if not query or not title:
            return 0.5

        # Simple word overlap scoring
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())

        if not query_words:
            return 0.5

        overlap = len(query_words & title_words)
        score = overlap / len(query_words)

        # Boost if query appears as substring
        if query.lower() in title.lower():
            score = min(1.0, score + 0.3)

        return min(1.0, max(0.3, score))

    async def _parse_results_with_llm(
        self,
        results: List[dict],
        query_properties: dict,
    ) -> List[dict]:
        """
        Parse results using LLM (slower but more accurate).
        Uses parallel async calls for speed.
        """
        if not results:
            return []

        # Create tasks for parallel execution
        tasks = [
            self._extract_product_from_result(result, query_properties)
            for result in results
        ]

        # Execute all LLM calls in parallel
        parsed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None results
        products = []
        for result in parsed_results:
            if isinstance(result, Exception):
                logger.warning(f"LLM parsing failed: {result}")
                continue
            if result is not None:
                products.append(result)

        return products

    async def _extract_product_from_result(
        self,
        result: dict,
        query_properties: dict,
    ) -> Optional[dict]:
        """
        Extract product information from a single search result.
        """
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        image_url = result.get("image_url")  # Get image from Tavily result

        if not title:
            return None

        # Use LLM to extract structured product info
        system_prompt = """Extract product information from search result.
Return JSON with product details or null if not a product."""

        prompt = f"""Search result:
Title: {title}
URL: {url}
Content: {content[:500]}

Original query looking for: {query_properties.get('name', '')}

Extract product info as JSON:
{{
    "name": "product name",
    "price": numeric_price_or_null,
    "currency": "USD",
    "merchant": "store name from URL",
    "source_url": "{url}",
    "match_confidence": 0.0 to 1.0 how well this matches the query
}}

Return null if this is not a product listing."""

        response = await self.call_llm(prompt, system_prompt)

        # Parse JSON response
        import json
        try:
            if "null" in response.lower():
                return None

            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                product = json.loads(response[start:end])
                product["match_source"] = "live_search"
                # Add image URL if available
                if image_url:
                    product["image_url"] = image_url
                return product

        except json.JSONDecodeError:
            pass

        # Fallback: basic extraction
        merchant = self._extract_merchant_from_url(url)
        return {
            "name": title,
            "source_url": url,
            "merchant": merchant,
            "image_url": image_url,  # Include image in fallback too
            "match_source": "live_search",
            "match_confidence": 0.5,
        }

    def _extract_merchant_from_url(self, url: str) -> str:
        """
        Extract merchant name from URL.
        """
        try:
            domain = urlparse(url).netloc
            # Remove www. and .com/.co.uk/etc
            parts = domain.replace("www.", "").split(".")
            return parts[0].capitalize()
        except Exception:
            return "Unknown"

    def _verify_matches(
        self,
        products: List[dict],
        query_properties: dict,
    ) -> List[dict]:
        """
        Verify and filter matches based on confidence.
        """
        verified = []

        for product in products:
            confidence = product.get("match_confidence", 0.5)

            # Only include confident matches
            if confidence >= 0.5:
                verified.append(product)

        # Sort by confidence
        verified.sort(key=lambda x: x.get("match_confidence", 0), reverse=True)

        return verified[:10]  # Top 10 results

    @traceable(name="search_product")
    async def search_product(
        self,
        product_name: str,
        brand: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[dict]:
        """
        Standalone method for searching a product.

        Can be used as a tool by other agents.
        """
        properties = {"name": product_name}
        if brand:
            properties["brand"] = brand
        if category:
            properties["category"] = category

        state = {
            "extracted_properties": properties,
        }

        result = await self._execute_impl(state)
        return result.get("live_search_results", [])

    def is_available(self) -> bool:
        """
        Check if live search is available.
        """
        return self.tavily_client is not None
