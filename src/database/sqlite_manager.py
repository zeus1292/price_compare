"""
SQLite database manager for product operations.
Handles CRUD operations and search queries.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

from sqlalchemy import and_, case, create_engine, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings
from src.database.models import (
    Base,
    IngestionLog,
    Product,
    ProductAttribute,
    SearchCache,
    SearchFeedback,
    SearchHistory,
    User,
    hash_password,
    init_db,
)


class SQLiteManager:
    """
    Manages SQLite database connections and operations.
    Supports both sync and async operations.
    """

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.sqlite_path
        self.sync_engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            pool_pre_ping=True,
        )
        self.async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )
        self.SyncSession = sessionmaker(bind=self.sync_engine)
        self.AsyncSession = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = False

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if not self._initialized:
            init_db(self.sync_engine)
            self._initialized = True

    @asynccontextmanager
    async def async_session(self):
        """Async context manager for database sessions."""
        async with self.AsyncSession() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def get_sync_session(self) -> Session:
        """Get a synchronous database session."""
        return self.SyncSession()

    # Product Operations

    async def create_product(self, product_data: dict) -> Product:
        """Create a new product in the database."""
        async with self.async_session() as session:
            # Normalize product name for search
            name = product_data.get("name", "")
            product_data["name_normalized"] = name.lower().strip()

            product = Product(**product_data)
            session.add(product)
            await session.commit()
            await session.refresh(product)
            return product

    async def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.id == product_id)
            )
            return result.scalar_one_or_none()

    async def get_product_by_gtin(self, gtin: str) -> Optional[Product]:
        """Get a product by GTIN (exact match)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.gtin == gtin)
            )
            return result.scalar_one_or_none()

    async def get_product_by_url(self, source_url: str) -> Optional[Product]:
        """Get a product by source URL."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.source_url == source_url)
            )
            return result.scalar_one_or_none()

    async def update_product(
        self,
        product_id: str,
        update_data: dict
    ) -> Optional[Product]:
        """Update an existing product."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.id == product_id)
            )
            product = result.scalar_one_or_none()
            if product:
                for key, value in update_data.items():
                    if hasattr(product, key):
                        setattr(product, key, value)
                product.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(product)
            return product

    async def delete_product(self, product_id: str) -> bool:
        """Delete a product by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.id == product_id)
            )
            product = result.scalar_one_or_none()
            if product:
                await session.delete(product)
                await session.commit()
                return True
            return False

    async def bulk_create_products(self, products_data: List[dict]) -> int:
        """Bulk create products for batch ingestion."""
        async with self.async_session() as session:
            products = []
            for data in products_data:
                name = data.get("name", "")
                data["name_normalized"] = name.lower().strip()
                products.append(Product(**data))
            session.add_all(products)
            await session.commit()
            return len(products)

    # Search Operations

    async def search_products(
        self,
        name_pattern: Optional[str] = None,
        merchant: Optional[str] = None,
        market: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Product]:
        """
        Search products with SQL filters.
        Returns candidates for further vector search refinement.
        """
        async with self.async_session() as session:
            query = select(Product)
            conditions = []

            if name_pattern:
                # Use normalized name for case-insensitive search
                pattern = f"%{name_pattern.lower().strip()}%"
                conditions.append(Product.name_normalized.like(pattern))

            if merchant:
                conditions.append(Product.merchant == merchant)

            if market:
                conditions.append(Product.market == market)

            if price_min is not None:
                conditions.append(Product.price >= price_min)

            if price_max is not None:
                conditions.append(Product.price <= price_max)

            if category:
                conditions.append(Product.category == category)

            if conditions:
                query = query.where(and_(*conditions))

            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_products_by_ids(self, product_ids: List[str]) -> List[Product]:
        """Get multiple products by their IDs."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Product).where(Product.id.in_(product_ids))
            )
            return list(result.scalars().all())

    async def count_products(
        self,
        merchant: Optional[str] = None,
        market: Optional[str] = None,
    ) -> int:
        """Count total products with optional filters."""
        async with self.async_session() as session:
            query = select(func.count(Product.id))
            conditions = []

            if merchant:
                conditions.append(Product.merchant == merchant)
            if market:
                conditions.append(Product.market == market)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            return result.scalar() or 0

    async def list_products(
        self,
        page: int = 1,
        limit: int = 20,
        merchant: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Tuple[List[Product], int]:
        """List products with pagination."""
        offset = (page - 1) * limit
        products = await self.search_products(
            merchant=merchant,
            market=market,
            limit=limit,
            offset=offset,
        )
        total = await self.count_products(merchant=merchant, market=market)
        return products, total

    # Cache Operations

    async def get_cached_search(self, query_hash: str) -> Optional[SearchCache]:
        """Get cached search result if not expired."""
        async with self.async_session() as session:
            result = await session.execute(
                select(SearchCache).where(
                    and_(
                        SearchCache.query_hash == query_hash,
                        or_(
                            SearchCache.expires_at.is_(None),
                            SearchCache.expires_at > datetime.utcnow()
                        )
                    )
                )
            )
            cache = result.scalar_one_or_none()
            if cache:
                cache.hit_count += 1
                await session.commit()
            return cache

    async def set_cached_search(
        self,
        query_hash: str,
        query_type: str,
        results_json: str,
        confidence_score: Optional[float] = None,
        ttl_hours: int = 24,
    ) -> SearchCache:
        """Store search result in cache."""
        async with self.async_session() as session:
            # Remove existing cache entry if any
            existing = await session.execute(
                select(SearchCache).where(SearchCache.query_hash == query_hash)
            )
            old_cache = existing.scalar_one_or_none()
            if old_cache:
                await session.delete(old_cache)

            cache = SearchCache(
                query_hash=query_hash,
                query_type=query_type,
                results_json=results_json,
                confidence_score=confidence_score,
                expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
            )
            session.add(cache)
            await session.commit()
            await session.refresh(cache)
            return cache

    async def clear_expired_cache(self) -> int:
        """Remove expired cache entries."""
        async with self.async_session() as session:
            result = await session.execute(
                select(SearchCache).where(
                    SearchCache.expires_at < datetime.utcnow()
                )
            )
            expired = result.scalars().all()
            for cache in expired:
                await session.delete(cache)
            await session.commit()
            return len(expired)

    # Ingestion Log Operations

    async def create_ingestion_log(self, file_path: str) -> IngestionLog:
        """Create a new ingestion log entry."""
        async with self.async_session() as session:
            log = IngestionLog(
                file_path=file_path,
                status="pending",
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log

    async def update_ingestion_log(
        self,
        log_id: int,
        status: str,
        records_processed: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[IngestionLog]:
        """Update an ingestion log entry."""
        async with self.async_session() as session:
            result = await session.execute(
                select(IngestionLog).where(IngestionLog.id == log_id)
            )
            log = result.scalar_one_or_none()
            if log:
                log.status = status
                if records_processed is not None:
                    log.records_processed = records_processed
                if error_message:
                    log.error_message = error_message
                if status == "processing" and not log.started_at:
                    log.started_at = datetime.utcnow()
                if status in ("completed", "failed"):
                    log.completed_at = datetime.utcnow()
                await session.commit()
                await session.refresh(log)
            return log

    async def get_pending_ingestion_files(self) -> List[str]:
        """Get list of files that haven't been processed yet."""
        async with self.async_session() as session:
            result = await session.execute(
                select(IngestionLog.file_path).where(
                    IngestionLog.status.in_(["pending", "failed"])
                )
            )
            return [row[0] for row in result.all()]

    # Feedback Operations

    async def create_feedback(
        self,
        query: str,
        query_type: str,
        rating: int,
        trace_id: Optional[str] = None,
        result_product_id: Optional[str] = None,
        result_name: Optional[str] = None,
        result_merchant: Optional[str] = None,
        result_confidence: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> SearchFeedback:
        """Create a new feedback entry."""
        async with self.async_session() as session:
            feedback = SearchFeedback(
                trace_id=trace_id,
                query=query,
                query_type=query_type,
                result_product_id=result_product_id,
                result_name=result_name,
                result_merchant=result_merchant,
                result_confidence=result_confidence,
                rating=rating,
                comment=comment,
            )
            session.add(feedback)
            await session.commit()
            await session.refresh(feedback)
            return feedback

    async def get_feedback(
        self,
        limit: int = 100,
        offset: int = 0,
        rating_filter: Optional[int] = None,
        query_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[SearchFeedback]:
        """Query feedback with optional filters."""
        async with self.async_session() as session:
            query = select(SearchFeedback)
            conditions = []

            if rating_filter is not None:
                conditions.append(SearchFeedback.rating == rating_filter)
            if query_type:
                conditions.append(SearchFeedback.query_type == query_type)
            if start_date:
                conditions.append(SearchFeedback.created_at >= start_date)
            if end_date:
                conditions.append(SearchFeedback.created_at <= end_date)

            if conditions:
                query = query.where(and_(*conditions))

            query = query.order_by(SearchFeedback.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_feedback_stats(self) -> dict:
        """Get feedback statistics for analysis."""
        async with self.async_session() as session:
            # Total counts
            total = await session.execute(
                select(func.count(SearchFeedback.id))
            )
            positive = await session.execute(
                select(func.count(SearchFeedback.id)).where(
                    SearchFeedback.rating == 1
                )
            )
            negative = await session.execute(
                select(func.count(SearchFeedback.id)).where(
                    SearchFeedback.rating == -1
                )
            )

            # By query type
            by_type = await session.execute(
                select(
                    SearchFeedback.query_type,
                    func.count(SearchFeedback.id).label("total"),
                    func.sum(
                        func.case((SearchFeedback.rating == 1, 1), else_=0)
                    ).label("positive"),
                    func.sum(
                        func.case((SearchFeedback.rating == -1, 1), else_=0)
                    ).label("negative"),
                ).group_by(SearchFeedback.query_type)
            )

            # Average confidence for positive vs negative
            avg_conf_positive = await session.execute(
                select(func.avg(SearchFeedback.result_confidence)).where(
                    SearchFeedback.rating == 1
                )
            )
            avg_conf_negative = await session.execute(
                select(func.avg(SearchFeedback.result_confidence)).where(
                    SearchFeedback.rating == -1
                )
            )

            total_count = total.scalar() or 0
            positive_count = positive.scalar() or 0
            negative_count = negative.scalar() or 0

            return {
                "total_feedback": total_count,
                "positive": positive_count,
                "negative": negative_count,
                "satisfaction_rate": (
                    positive_count / total_count if total_count > 0 else 0.0
                ),
                "by_query_type": [
                    {
                        "type": row.query_type,
                        "total": row.total,
                        "positive": row.positive or 0,
                        "negative": row.negative or 0,
                    }
                    for row in by_type.all()
                ],
                "avg_confidence_positive": avg_conf_positive.scalar(),
                "avg_confidence_negative": avg_conf_negative.scalar(),
            }

    async def get_feedback_by_trace(self, trace_id: str) -> List[SearchFeedback]:
        """Get all feedback for a specific trace."""
        async with self.async_session() as session:
            result = await session.execute(
                select(SearchFeedback).where(
                    SearchFeedback.trace_id == trace_id
                )
            )
            return list(result.scalars().all())

    # Statistics

    async def get_stats(self) -> dict:
        """Get database statistics."""
        async with self.async_session() as session:
            product_count = await session.execute(
                select(func.count(Product.id))
            )
            cache_count = await session.execute(
                select(func.count(SearchCache.id))
            )
            merchants = await session.execute(
                select(func.count(func.distinct(Product.merchant)))
            )
            markets = await session.execute(
                select(func.count(func.distinct(Product.market)))
            )

            return {
                "total_products": product_count.scalar() or 0,
                "cached_searches": cache_count.scalar() or 0,
                "unique_merchants": merchants.scalar() or 0,
                "unique_markets": markets.scalar() or 0,
            }

    # User Operations

    async def create_user(self, email: str, password: str, name: Optional[str] = None) -> User:
        """Create a new user account."""
        async with self.async_session() as session:
            user = User(
                email=email.lower().strip(),
                password_hash=hash_password(password),
                name=name,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.email == email.lower().strip())
            )
            return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()

    async def verify_user(self, email: str, password: str) -> Optional[User]:
        """Verify user credentials and return user if valid."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(
                    and_(
                        User.email == email.lower().strip(),
                        User.password_hash == hash_password(password),
                        User.is_active == True
                    )
                )
            )
            user = result.scalar_one_or_none()
            if user:
                # Update last login
                user.last_login = datetime.utcnow()
                await session.commit()
            return user

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.last_login = datetime.utcnow()
                await session.commit()

    # Search History Operations

    async def add_search_history(
        self,
        user_id: str,
        query: str,
        query_type: str,
        result_count: int = 0
    ) -> SearchHistory:
        """Add a search to user's history."""
        async with self.async_session() as session:
            history = SearchHistory(
                user_id=user_id,
                query=query,
                query_type=query_type,
                result_count=result_count,
            )
            session.add(history)
            await session.commit()
            await session.refresh(history)
            return history

    async def get_recent_searches(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[SearchHistory]:
        """Get user's recent search history."""
        async with self.async_session() as session:
            result = await session.execute(
                select(SearchHistory)
                .where(SearchHistory.user_id == user_id)
                .order_by(SearchHistory.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def clear_search_history(self, user_id: str) -> int:
        """Clear all search history for a user. Returns count of deleted records."""
        async with self.async_session() as session:
            result = await session.execute(
                select(SearchHistory).where(SearchHistory.user_id == user_id)
            )
            histories = result.scalars().all()
            count = len(histories)
            for history in histories:
                await session.delete(history)
            await session.commit()
            return count

    def close(self) -> None:
        """Close database connections."""
        self.sync_engine.dispose()

    async def close_async(self) -> None:
        """Close async database connections."""
        await self.async_engine.dispose()
