"""
Google Trends refresh service.

Fetches live category popularity scores from Google Trends on startup
and then every `interval_hours` hours. Falls back gracefully to the
existing static scores when Trends is unavailable or rate-limited.
"""
import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.services.category_service import CategoryService

logger = logging.getLogger(__name__)

# Maps category IDs → representative search terms for Google Trends
CATEGORY_TRENDS_KEYWORDS = {
    "electronics": "electronics",
    "fashion": "fashion",
    "home_kitchen": "home kitchen",
    "beauty": "beauty skincare",
    "toys_games": "toys games",
}


class TrendsRefreshService:
    """
    Background service that refreshes category popularity scores from Google Trends.

    Runs once immediately on startup, then every `interval_hours` hours.
    Falls back to static scores if Google Trends is unavailable.
    """

    def __init__(self, category_service: "CategoryService", interval_hours: int = 24):
        self._category_service = category_service
        self._interval_seconds = interval_hours * 3600
        self._task: Optional[asyncio.Task] = None
        self.last_updated: Optional[datetime] = None
        self.last_scores: dict = {}

    async def start(self) -> None:
        """Start the background refresh loop."""
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"TrendsRefreshService started (interval={self._interval_seconds // 3600}h)"
        )

    async def stop(self) -> None:
        """Cancel the background task on shutdown."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TrendsRefreshService stopped")

    async def _refresh_loop(self) -> None:
        """Refresh immediately on startup, then sleep for the configured interval."""
        while True:
            try:
                await self._refresh()
            except Exception as e:
                logger.error(f"Trends refresh error: {e}", exc_info=True)
            await asyncio.sleep(self._interval_seconds)

    async def _refresh(self) -> None:
        """Fetch Google Trends data and push updated scores to CategoryService."""
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._fetch_trends_sync)
        if scores:
            self._category_service.update_popularity_scores(scores)
            self.last_updated = datetime.utcnow()
            self.last_scores = scores
            logger.info(f"Category popularity updated from Google Trends: {scores}")
        else:
            logger.warning("Google Trends returned no data — keeping existing scores")

    def _fetch_trends_sync(self) -> dict:
        """
        Synchronous pytrends fetch — called via run_in_executor so it doesn't
        block the event loop.

        Returns a dict of {category_id: score (0-100)}, or {} on failure.
        """
        try:
            from pytrends.request import TrendReq  # lazy import

            pytrends = TrendReq(
                hl="en-US",
                tz=0,
                timeout=(10, 30),
                retries=2,
                backoff_factor=0.5,
            )
            keywords = list(CATEGORY_TRENDS_KEYWORDS.values())
            pytrends.build_payload(keywords, timeframe="now 7-d", geo="US")
            df = pytrends.interest_over_time()

            if df.empty:
                logger.warning("pytrends returned an empty dataframe")
                return {}

            raw: dict = {}
            for cat_id, keyword in CATEGORY_TRENDS_KEYWORDS.items():
                if keyword in df.columns:
                    raw[cat_id] = float(df[keyword].mean())

            if not raw:
                return {}

            max_score = max(raw.values())
            if max_score == 0:
                return {}

            # Normalize so the top category always scores 100
            return {cat_id: round(score / max_score * 100) for cat_id, score in raw.items()}

        except Exception as e:
            logger.error(f"pytrends fetch failed: {e}")
            return {}

    def get_status(self) -> dict:
        """Return current service status (for health checks)."""
        return {
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "last_scores": self.last_scores,
            "interval_hours": self._interval_seconds // 3600,
        }


# Singleton
_trends_service: Optional[TrendsRefreshService] = None


def get_trends_service() -> Optional[TrendsRefreshService]:
    """Return the running trends service instance (set during app lifespan)."""
    return _trends_service


def set_trends_service(service: TrendsRefreshService) -> None:
    """Register the singleton (called from app lifespan)."""
    global _trends_service
    _trends_service = service
