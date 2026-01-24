"""
Product CRUD API routes.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.database.sqlite_manager import SQLiteManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ProductBase(BaseModel):
    """Base product model."""
    name: str
    price: Optional[float] = None
    currency: str = "USD"
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    merchant: Optional[str] = None
    market: Optional[str] = None
    category: Optional[str] = None
    gtin: Optional[str] = None


class ProductCreate(ProductBase):
    """Product creation model."""
    pass


class ProductUpdate(BaseModel):
    """Product update model."""
    name: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    merchant: Optional[str] = None
    market: Optional[str] = None
    category: Optional[str] = None
    gtin: Optional[str] = None


class ProductResponse(ProductBase):
    """Product response model."""
    id: str
    name_normalized: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProductListResponse(BaseModel):
    """Product list response."""
    products: List[dict]
    total: int
    page: int
    limit: int
    pages: int


def get_db() -> SQLiteManager:
    """Get database manager."""
    db = SQLiteManager()
    db.initialize()
    return db


@router.get("/products/{product_id}")
async def get_product(product_id: str):
    """
    Get a product by ID.
    """
    db = get_db()

    try:
        product = await db.get_product(product_id)

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        return product.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    merchant: Optional[str] = Query(None, description="Filter by merchant"),
    market: Optional[str] = Query(None, description="Filter by market"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """
    List products with pagination.
    """
    db = get_db()

    try:
        products, total = await db.list_products(
            page=page,
            limit=limit,
            merchant=merchant,
            market=market,
        )

        pages = (total + limit - 1) // limit

        return ProductListResponse(
            products=[p.to_dict() for p in products],
            total=total,
            page=page,
            limit=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error(f"Failed to list products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/products")
async def create_product(product: ProductCreate):
    """
    Create a new product.
    """
    db = get_db()

    try:
        created = await db.create_product(product.model_dump())
        return created.to_dict()

    except Exception as e:
        logger.error(f"Failed to create product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/products/{product_id}")
async def update_product(product_id: str, product: ProductUpdate):
    """
    Update an existing product.
    """
    db = get_db()

    try:
        # Filter out None values
        update_data = {k: v for k, v in product.model_dump().items() if v is not None}

        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")

        updated = await db.update_product(product_id, update_data)

        if not updated:
            raise HTTPException(status_code=404, detail="Product not found")

        return updated.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/products/{product_id}")
async def delete_product(product_id: str):
    """
    Delete a product.
    """
    db = get_db()

    try:
        deleted = await db.delete_product(product_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Product not found")

        # Also delete from ChromaDB
        from src.database.chroma_manager import ChromaManager
        chroma = ChromaManager()
        chroma.delete_product(product_id)

        return {"message": "Product deleted", "id": product_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/gtin/{gtin}")
async def get_product_by_gtin(gtin: str):
    """
    Get a product by GTIN.
    """
    db = get_db()

    try:
        product = await db.get_product_by_gtin(gtin)

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        return product.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get product by GTIN {gtin}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ContributeProductRequest(BaseModel):
    """Request to contribute a product to the database."""
    name: str = Field(..., min_length=3, description="Product name")
    price: Optional[float] = Field(None, ge=0, description="Product price")
    currency: str = Field(default="USD", description="Currency code")
    image_url: Optional[str] = Field(None, description="Product image URL")
    source_url: Optional[str] = Field(None, description="Product page URL")
    merchant: Optional[str] = Field(None, description="Merchant/store name")
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Brand name")


@router.post("/products/contribute")
async def contribute_product(request: ContributeProductRequest):
    """
    Contribute a product to the database.

    This endpoint allows users to add products they've searched for,
    improving the database for future searches. Products are indexed
    for both SQL and vector search.
    """
    from src.database.chroma_manager import ChromaManager
    from src.services.embedding_service import EmbeddingService

    db = get_db()
    chroma = ChromaManager()
    embeddings = EmbeddingService()

    try:
        # Check if product already exists by URL
        if request.source_url:
            existing = await db.get_product_by_url(request.source_url)
            if existing:
                return {
                    "message": "Product already exists",
                    "product": existing.to_dict(),
                    "created": False,
                }

        # Create product in SQLite
        product_data = request.model_dump()
        product_data["name_normalized"] = request.name.lower().strip()

        created = await db.create_product(product_data)
        product_dict = created.to_dict()

        # Generate and store embedding
        try:
            embedding = await embeddings.embed_text_async(request.name)
            chroma.add_product_embedding(
                product_id=str(created.id),
                name=request.name,
                embedding=embedding,
                metadata={
                    "merchant": request.merchant or "",
                    "category": request.category or "",
                    "price": request.price or 0.0,
                },
            )
            product_dict["embedding_generated"] = True
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            product_dict["embedding_generated"] = False

        return {
            "message": "Product contributed successfully",
            "product": product_dict,
            "created": True,
        }

    except Exception as e:
        logger.error(f"Failed to contribute product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/stats")
async def get_product_stats():
    """
    Get database statistics.
    """
    from src.database.chroma_manager import ChromaManager

    db = get_db()
    chroma = ChromaManager()

    try:
        db_stats = await db.get_stats()
        chroma_stats = chroma.get_stats()

        return {
            "database": db_stats,
            "vector_store": chroma_stats,
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
