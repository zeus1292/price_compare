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
