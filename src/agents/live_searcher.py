"""
Live Search Agent using Tavily for web search.
"""
import logging
import time
from typing import List, Optional

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
        start_time = time.perf_counter()

        try:
            # Build search query
            search_query = self._build_search_query(properties)
            logger.info(f"Live search query: {search_query}")

            # Execute Tavily search
            tavily_results = await self._tavily_search(search_query)

            # Parse and extract product info from results
            parsed_products = await self._parse_search_results(
                tavily_results, properties
            )

            # Verify matches
            verified = self._verify_matches(parsed_products, properties)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

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
        max_results: int = 10,
    ) -> List[dict]:
        """
        Execute Tavily search.
        """
        if not self.tavily_client:
            return []

        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_images=True,  # Include images in results
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

    @traceable(name="parse_search_results")
    async def _parse_search_results(
        self,
        results: List[dict],
        query_properties: dict,
    ) -> List[dict]:
        """
        Parse Tavily results and extract product information.
        """
        if not results:
            return []

        parsed_products = []

        for result in results:
            try:
                product = await self._extract_product_from_result(
                    result, query_properties
                )
                if product:
                    parsed_products.append(product)

            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        return parsed_products

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
        from urllib.parse import urlparse

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
