"""
SQLAlchemy ORM models for the product database.
"""
from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
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


def init_db(engine) -> None:
    """Initialize database tables."""
    Base.metadata.create_all(engine)
