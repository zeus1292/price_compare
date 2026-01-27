"""
Search Metrics Service for tracking precision, recall, and performance.

Since we don't have ground truth labels, we use proxy metrics:
- Precision proxy: % of results above confidence threshold
- Recall proxy: Whether any results were found for the query
- Coverage: Distribution of match sources (DB vs live)
"""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics for a single search."""
    query: str
    input_type: str
    total_results: int
    high_confidence_results: int  # >= 0.8
    medium_confidence_results: int  # 0.5 - 0.8
    low_confidence_results: int  # < 0.5
    db_results: int
    live_results: int
    live_search_triggered: bool
    processing_time_ms: int
    extraction_confidence: float
    avg_result_confidence: float

    # Proxy metrics
    precision_proxy: float  # high_confidence / total (if total > 0)
    recall_proxy: float  # 1.0 if results found, 0.0 otherwise

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:50] + "..." if len(self.query) > 50 else self.query,
            "input_type": self.input_type,
            "total_results": self.total_results,
            "confidence_distribution": {
                "high": self.high_confidence_results,
                "medium": self.medium_confidence_results,
                "low": self.low_confidence_results,
            },
            "source_distribution": {
                "database": self.db_results,
                "live": self.live_results,
            },
            "live_search_triggered": self.live_search_triggered,
            "processing_time_ms": self.processing_time_ms,
            "extraction_confidence": round(self.extraction_confidence, 3),
            "avg_result_confidence": round(self.avg_result_confidence, 3),
            "precision_proxy": round(self.precision_proxy, 3),
            "recall_proxy": round(self.recall_proxy, 3),
        }


class MetricsService:
    """
    Service for collecting and reporting search metrics.

    Tracks:
    - Per-search metrics
    - Aggregate statistics
    - Performance trends
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.search_history: List[SearchMetrics] = []
        self.aggregate_stats: Dict[str, Any] = defaultdict(lambda: defaultdict(float))

    def record_search(
        self,
        query: str,
        input_type: str,
        results: List[dict],
        live_search_triggered: bool,
        processing_time_ms: int,
        extraction_confidence: float = 0.0,
    ) -> SearchMetrics:
        """
        Record metrics for a completed search.

        Args:
            query: Original search query
            input_type: text, url, or image
            results: List of result dicts with match_confidence and match_source
            live_search_triggered: Whether live search was used
            processing_time_ms: Total processing time
            extraction_confidence: Confidence from property extraction

        Returns:
            SearchMetrics object
        """
        # Count by confidence level
        high_conf = sum(1 for r in results if r.get("match_confidence", 0) >= 0.8)
        medium_conf = sum(1 for r in results if 0.5 <= r.get("match_confidence", 0) < 0.8)
        low_conf = sum(1 for r in results if r.get("match_confidence", 0) < 0.5)

        # Count by source
        db_results = sum(1 for r in results if r.get("match_source") != "live")
        live_results = sum(1 for r in results if r.get("match_source") == "live")

        # Calculate average confidence
        total = len(results)
        avg_conf = (
            sum(r.get("match_confidence", 0) for r in results) / total
            if total > 0 else 0.0
        )

        # Proxy metrics
        precision_proxy = high_conf / total if total > 0 else 0.0
        recall_proxy = 1.0 if total > 0 else 0.0

        metrics = SearchMetrics(
            query=query,
            input_type=input_type,
            total_results=total,
            high_confidence_results=high_conf,
            medium_confidence_results=medium_conf,
            low_confidence_results=low_conf,
            db_results=db_results,
            live_results=live_results,
            live_search_triggered=live_search_triggered,
            processing_time_ms=processing_time_ms,
            extraction_confidence=extraction_confidence,
            avg_result_confidence=avg_conf,
            precision_proxy=precision_proxy,
            recall_proxy=recall_proxy,
        )

        # Store in history
        self.search_history.append(metrics)
        if len(self.search_history) > self.max_history:
            self.search_history.pop(0)

        # Update aggregates
        self._update_aggregates(metrics)

        logger.info(
            f"Search metrics: precision={precision_proxy:.2f}, "
            f"recall={recall_proxy:.2f}, results={total}, "
            f"time={processing_time_ms}ms"
        )

        return metrics

    def _update_aggregates(self, metrics: SearchMetrics) -> None:
        """Update running aggregate statistics."""
        key = metrics.input_type
        stats = self.aggregate_stats[key]

        stats["search_count"] += 1
        stats["total_results"] += metrics.total_results
        stats["total_high_conf"] += metrics.high_confidence_results
        stats["total_db_results"] += metrics.db_results
        stats["total_live_results"] += metrics.live_results
        stats["total_time_ms"] += metrics.processing_time_ms
        stats["live_search_count"] += 1 if metrics.live_search_triggered else 0
        stats["precision_sum"] += metrics.precision_proxy
        stats["recall_sum"] += metrics.recall_proxy

    def get_aggregate_report(self) -> Dict[str, Any]:
        """
        Get aggregate metrics report.

        Returns:
            Dictionary with aggregate statistics by input type
        """
        report = {
            "total_searches": len(self.search_history),
            "by_input_type": {},
        }

        for input_type, stats in self.aggregate_stats.items():
            count = stats["search_count"]
            if count == 0:
                continue

            report["by_input_type"][input_type] = {
                "search_count": int(count),
                "avg_results_per_search": round(stats["total_results"] / count, 2),
                "avg_high_confidence_results": round(stats["total_high_conf"] / count, 2),
                "avg_processing_time_ms": round(stats["total_time_ms"] / count, 0),
                "live_search_rate": round(stats["live_search_count"] / count, 3),
                "db_vs_live_ratio": round(
                    stats["total_db_results"] / max(1, stats["total_live_results"]), 2
                ),
                "avg_precision_proxy": round(stats["precision_sum"] / count, 3),
                "avg_recall_proxy": round(stats["recall_sum"] / count, 3),
            }

        # Overall averages
        total_count = len(self.search_history)
        if total_count > 0:
            report["overall"] = {
                "avg_precision_proxy": round(
                    sum(m.precision_proxy for m in self.search_history) / total_count, 3
                ),
                "avg_recall_proxy": round(
                    sum(m.recall_proxy for m in self.search_history) / total_count, 3
                ),
                "avg_processing_time_ms": round(
                    sum(m.processing_time_ms for m in self.search_history) / total_count, 0
                ),
                "avg_results_per_search": round(
                    sum(m.total_results for m in self.search_history) / total_count, 2
                ),
            }

        return report

    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get metrics for recent searches."""
        return [m.to_dict() for m in self.search_history[-limit:]]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a performance summary with recommendations.
        """
        if not self.search_history:
            return {"message": "No searches recorded yet"}

        recent = self.search_history[-100:]  # Last 100 searches

        avg_precision = sum(m.precision_proxy for m in recent) / len(recent)
        avg_recall = sum(m.recall_proxy for m in recent) / len(recent)
        avg_time = sum(m.processing_time_ms for m in recent) / len(recent)
        live_rate = sum(1 for m in recent if m.live_search_triggered) / len(recent)

        recommendations = []

        if avg_precision < 0.5:
            recommendations.append(
                "Low precision: Consider improving property extraction or "
                "adjusting confidence scoring"
            )

        if avg_recall < 0.8:
            recommendations.append(
                "Low recall: Consider expanding database coverage or "
                "lowering confidence thresholds"
            )

        if live_rate > 0.7:
            recommendations.append(
                "High live search rate: Consider expanding local database "
                "to reduce API costs and latency"
            )

        if avg_time > 5000:
            recommendations.append(
                "High latency: Consider enabling fast_mode or caching frequent queries"
            )

        return {
            "sample_size": len(recent),
            "metrics": {
                "avg_precision_proxy": round(avg_precision, 3),
                "avg_recall_proxy": round(avg_recall, 3),
                "avg_processing_time_ms": round(avg_time, 0),
                "live_search_rate": round(live_rate, 3),
            },
            "health": {
                "precision": "good" if avg_precision >= 0.6 else "needs_improvement",
                "recall": "good" if avg_recall >= 0.8 else "needs_improvement",
                "latency": "good" if avg_time < 3000 else "needs_improvement",
            },
            "recommendations": recommendations if recommendations else ["All metrics look healthy!"],
        }


# Singleton instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get or create the metrics service singleton."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
