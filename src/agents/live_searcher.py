"""
Live Search Agent using Tavily for web search.

Optimized for speed with:
- Parallel LLM calls for result parsing
- Basic search depth option for faster results
- Heuristic-based extraction as fallback
- Result caching to avoid redundant API calls
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse

from langsmith import traceable
from tavily import TavilyClient

from src.agents.base_agent import BaseAgent
from src.config.settings import get_settings
from src.services.llm_service import LLMService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilyCache:
    """
    In-memory cache for Tavily search results.

    Caches results for a configurable TTL to avoid redundant API calls.
    """

    def __init__(self, ttl_minutes: int = 30, max_size: int = 1000):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = max_size
        self._cache: Dict[str, dict] = {}

    def _hash_query(self, query: str, search_depth: str) -> str:
        """Generate cache key from query and settings."""
        key = f"{query}:{search_depth}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, query: str, search_depth: str) -> Optional[List[dict]]:
        """Get cached results if available and not expired."""
        cache_key = self._hash_query(query, search_depth)

        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() < entry["expires_at"]:
                logger.info(f"Tavily cache hit for query: {query[:50]}...")
                return entry["results"]
            else:
                # Expired - remove from cache
                del self._cache[cache_key]

        return None

    def set(self, query: str, search_depth: str, results: List[dict]) -> None:
        """Cache search results."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["expires_at"]
            )
            del self._cache[oldest_key]

        cache_key = self._hash_query(query, search_depth)
        self._cache[cache_key] = {
            "results": results,
            "expires_at": datetime.now() + self.ttl,
            "cached_at": datetime.now(),
        }
        logger.info(f"Tavily result cached for query: {query[:50]}...")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(1 for e in self._cache.values() if now < e["expires_at"])
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


# Singleton cache instance
_tavily_cache: Optional[TavilyCache] = None


def get_tavily_cache() -> TavilyCache:
    """Get or create the Tavily cache singleton."""
    global _tavily_cache
    if _tavily_cache is None:
        _tavily_cache = TavilyCache()
    return _tavily_cache


class LiveSearcherAgent(BaseAgent):
    """
    Agent for searching the web when database has no confident match.

    Uses Tavily API for web search and LLM for result parsing.
    Results are cached to avoid redundant API calls.

    Output:
        {
            "live_search_results": [...],
            "live_search_triggered": True,
            "search_time_ms": 1234,
            "cache_hit": True/False
        }
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        tavily_api_key: Optional[str] = None,
        cache: Optional[TavilyCache] = None,
    ):
        super().__init__(name="LiveSearcher", llm_service=llm_service)

        settings = get_settings()
        self.tavily_key = tavily_api_key or settings.tavily_api_key.get_secret_value()
        self.cache = cache or get_tavily_cache()

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

            # Execute Tavily search
            # Use advanced depth for better product page results
            search_depth = "basic" if fast_mode else "advanced"
            tavily_results = await self._tavily_search(
                search_query,
                max_results=15,  # More results since we filter out category pages
                search_depth=search_depth,
            )

            # Parse results - fast mode uses heuristics, slow mode uses LLM
            results_list = tavily_results.get("results", [])
            if fast_mode:
                parsed_products = self._parse_results_fast(results_list, properties)
            else:
                parsed_products = await self._parse_results_with_llm(
                    results_list, properties
                )

            # Verify matches
            verified = self._verify_matches(parsed_products, properties)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"Live search completed in {elapsed_ms}ms (fast_mode={fast_mode})")

            return self.update_state(state, {
                "live_search_results": verified,
                "live_search_triggered": True,
                "live_search_query": search_query,
                "live_search_raw_count": len(tavily_results.get("results", [])),
                "live_search_cache_hit": tavily_results.get("cache_hit", False),
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

        # Add "buy" to get shopping/product results (not reviews/news)
        parts.append("buy")

        return " ".join(parts)

    def _is_product_page(self, url: str) -> bool:
        """
        Check if URL is likely a product page vs category/search page.
        """
        url_lower = url.lower()

        # Patterns indicating category/search pages (NOT product pages)
        category_patterns = [
            "/browse/",
            "/search",
            "/searchpage",
            "/category/",
            "/categories/",
            "/shop/",
            "/c/",
            "/s?k=",  # Amazon search
            "/b/",  # eBay browse
            "/bn_",  # eBay browse node
            "?q=",
            "?query=",
            "?search=",
            "/collection/",
            "/collections/",
            "/stores/",  # Amazon store pages
            "/page/",  # Amazon brand pages
            "/pcmcat",  # Best Buy category pages
            "/site/searchpage",
        ]

        for pattern in category_patterns:
            if pattern in url_lower:
                return False

        # Patterns indicating product pages
        product_patterns = [
            "/dp/",  # Amazon product
            "/ip/",  # Walmart product
            "/p/",   # Various retailers product
            "/product/",
            "/products/",
            "/item/",
            "/itm/",  # eBay item
            "/pd/",  # Best Buy product detail
            "/sku/",
            "/gp/product/",  # Amazon alternate
        ]

        for pattern in product_patterns:
            if pattern in url_lower:
                return True

        # Default: reject if no product pattern matched
        # This is more conservative but gives better quality results
        return False

    @traceable(name="tavily_search")
    async def _tavily_search(
        self,
        query: str,
        max_results: int = 8,
        search_depth: str = "basic",
        use_cache: bool = True,
    ) -> dict:
        """
        Execute Tavily search with caching.

        Args:
            query: Search query
            max_results: Maximum results to return
            search_depth: "basic" (fast) or "advanced" (thorough)
            use_cache: Whether to use/update cache

        Returns:
            Dict with "results" list and "cache_hit" boolean
        """
        if not self.tavily_client:
            return {"results": [], "cache_hit": False}

        # Check cache first
        if use_cache:
            cached_results = self.cache.get(query, search_depth)
            if cached_results is not None:
                return {"results": cached_results, "cache_hit": True}

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

            # Attach per-result images first (Tavily newer API puts image on each result)
            # Fall back to the global images list only for results that have no image
            imageless_indices = []
            for i, result in enumerate(results):
                if result.get("image"):
                    result["image_url"] = result["image"]
                else:
                    imageless_indices.append(i)

            # Assign spare global images to imageless results in order
            for spare_url, idx in zip(images, imageless_indices):
                if spare_url:
                    results[idx]["image_url"] = spare_url

            # Cache results
            if use_cache:
                self.cache.set(query, search_depth, results)

            return {"results": results, "cache_hit": False}

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {"results": [], "cache_hit": False}

    def _parse_results_fast(
        self,
        results: List[dict],
        query_properties: dict,
    ) -> List[dict]:
        """
        Fast parsing using heuristics instead of LLM.
        Extracts product info from title, URL, and content using regex.
        Filters out category/search pages to only return actual product pages.
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

            # Skip category/search pages - only keep actual product pages
            if not self._is_product_page(url):
                logger.debug(f"Skipping non-product URL: {url}")
                continue

            # Extract price using regex - try content first (more specific)
            price = self._extract_price_fast(content)
            if price is None:
                price = self._extract_price_fast(title)

            # Extract merchant from URL
            merchant = self._extract_merchant_from_url(url)

            # Clean up title - remove store name suffixes
            clean_title = self._clean_product_title(title, merchant)

            # Calculate match confidence based on title similarity
            confidence = self._calculate_similarity(query_name, clean_title.lower())

            # Boost confidence if we found a price (indicates real product page)
            if price is not None:
                confidence = min(1.0, confidence + 0.1)

            parsed.append({
                "name": clean_title,
                "price": price,
                "currency": "USD",
                "merchant": merchant,
                "source_url": url,
                "image_url": image_url,
                "match_source": "live",
                "match_confidence": confidence,
            })

        return parsed

    def _clean_product_title(self, title: str, merchant: str) -> str:
        """
        Clean product title by removing store name suffixes.
        """
        # Common patterns to remove
        patterns_to_remove = [
            r"\s*[-|]\s*(Amazon|Walmart|Best Buy|Target|eBay|Newegg).*$",
            r"\s*:\s*(Amazon|Walmart|Best Buy|Target|eBay|Newegg).*$",
            r"\s*-\s*[A-Za-z]+\.com$",
            r"\s*\|\s*[A-Za-z]+\.com$",
        ]

        cleaned = title
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _extract_price_fast(self, text: str) -> Optional[float]:
        """
        Extract price from text using regex.
        Returns the first reasonable price found.
        """
        if not text:
            return None

        # Match patterns like $99.99, $1,234.56, USD 99.99
        patterns = [
            r'\$\s*([0-9,]+\.?\d{0,2})',  # $99.99 or $ 99.99
            r'USD\s*([0-9,]+\.?\d{0,2})',  # USD 99.99
            r'Price[:\s]+\$?\s*([0-9,]+\.?\d{0,2})',  # Price: $99.99
            r'Now\s*\$\s*([0-9,]+\.?\d{0,2})',  # Now $99.99
            r'Sale\s*\$\s*([0-9,]+\.?\d{0,2})',  # Sale $99.99
            r'(?:^|\s)\$([0-9,]+\.?\d{0,2})(?:\s|$)',  # Standalone $99.99
        ]

        prices_found = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price_str = match.replace(',', '').strip()
                    if price_str:
                        price = float(price_str)
                        # Sanity check: reasonable product price range
                        if 1.0 <= price <= 50000.0:
                            prices_found.append(price)
                except ValueError:
                    continue

        # Return the most likely price (often the first one found)
        if prices_found:
            return prices_found[0]

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

    def get_cache_stats(self) -> dict:
        """
        Get Tavily cache statistics.
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """
        Clear the Tavily cache.
        """
        self.cache.clear()
        logger.info("Tavily cache cleared")
