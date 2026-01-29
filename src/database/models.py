"""
SQLAlchemy ORM models for the product database.
"""
from datetime import datetime
from typing import Optional
import uuid
import hashlib

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = "pricehawk_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


class Product(Base):
    """
    Core product information table.
    Stores extracted product data from various sources.
    """
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(Text, nullable=False)
    name_normalized = Column(Text, nullable=False, index=True)
    price = Column(Float, nullable=True)
    currency = Column(String(10), default="USD")
    image_url = Column(Text, nullable=True)
    image_hash = Column(String(64), nullable=True)
    source_url = Column(Text, nullable=True, unique=True)
    merchant = Column(String(255), nullable=True, index=True)
    market = Column(String(10), nullable=True, index=True)
    category = Column(String(255), nullable=True)
    gtin = Column(String(14), nullable=True, index=True)
    embedding_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    attributes = relationship(
        "ProductAttribute",
        back_populates="product",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_products_price", "price"),
        Index("idx_products_category", "category"),
    )

    def to_dict(self) -> dict:
        """Convert product to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "name_normalized": self.name_normalized,
            "price": self.price,
            "currency": self.currency,
            "image_url": self.image_url,
            "source_url": self.source_url,
            "merchant": self.merchant,
            "market": self.market,
            "category": self.category,
            "gtin": self.gtin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProductAttribute(Base):
    """
    Flexible key-value attributes for products.
    Allows storing additional metadata without schema changes.
    """
    __tablename__ = "product_attributes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(
        String(36),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False
    )
    attribute_key = Column(String(255), nullable=False)
    attribute_value = Column(Text, nullable=True)

    # Relationships
    product = relationship("Product", back_populates="attributes")

    __table_args__ = (
        Index("idx_attributes_product", "product_id"),
        Index("idx_attributes_key_value", "attribute_key", "attribute_value"),
    )


class SearchCache(Base):
    """
    Cache for expensive search results.
    Stores serialized results with expiration.
    """
    __tablename__ = "search_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_type = Column(String(20), nullable=False)  # 'text', 'image', 'url'
    results_json = Column(Text, nullable=False)  # Encrypted JSON
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)
    hit_count = Column(Integer, default=0)


class IngestionLog(Base):
    """
    Track dataset processing progress.
    Useful for resuming interrupted ingestion.
    """
    __tablename__ = "ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(Text, nullable=False)
    status = Column(String(20), nullable=False)  # 'pending', 'processing', 'completed', 'failed'
    records_processed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_ingestion_status", "status"),
    )


class SearchFeedback(Base):
    """
    User feedback on search results.
    Stores thumbs up/down ratings with context for analysis.
    """
    __tablename__ = "search_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String(36), nullable=True, index=True)  # Links to LangSmith trace
    query = Column(Text, nullable=False)
    query_type = Column(String(20), nullable=False)  # 'text', 'url', 'image'
    result_product_id = Column(String(36), nullable=True)  # Product that was rated
    result_name = Column(Text, nullable=True)  # Product name for quick reference
    result_merchant = Column(String(255), nullable=True)
    result_confidence = Column(Float, nullable=True)  # Confidence score when shown
    rating = Column(Integer, nullable=False)  # 1 = thumbs up, -1 = thumbs down
    comment = Column(Text, nullable=True)  # Optional user comment
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_feedback_rating", "rating"),
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_query_type", "query_type"),
    )

    def to_dict(self) -> dict:
        """Convert feedback to dictionary."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "query": self.query,
            "query_type": self.query_type,
            "result_product_id": self.result_product_id,
            "result_name": self.result_name,
            "result_merchant": self.result_merchant,
            "result_confidence": self.result_confidence,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """
    User account for authentication and personalization.
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(64), nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    search_history = relationship(
        "SearchHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(SearchHistory.created_at)"
    )

    __table_args__ = (
        Index("idx_users_email", "email"),
    )

    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding password)."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class SearchHistory(Base):
    """
    Track user search history for personalization.
    """
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    query = Column(Text, nullable=False)
    query_type = Column(String(20), nullable=False)  # 'text', 'url', 'image'
    result_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="search_history")

    __table_args__ = (
        Index("idx_search_history_user", "user_id"),
        Index("idx_search_history_created", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert search history to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "query_type": self.query_type,
            "result_count": self.result_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db(engine) -> None:
    """Initialize database tables."""
    Base.metadata.create_all(engine)
