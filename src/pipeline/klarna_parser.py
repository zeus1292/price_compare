"""
Parser for Klarna product dataset files.

Supports MHTML, WTL snapshot formats with klarna-ai-label extraction.
"""
import json
import os
import re
from dataclasses import dataclass, field
from email import message_from_bytes, message_from_file
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup


@dataclass
class ParsedProduct:
    """Parsed product data from Klarna dataset."""
    name: str = ""
    price: Optional[float] = None
    currency: str = "USD"
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    merchant: Optional[str] = None
    market: Optional[str] = None
    raw_price: Optional[str] = None
    file_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class KlarnaParser:
    """
    Parser for Klarna product pages.

    Extracts product information using klarna-ai-label attributes:
    - "Price" -> price
    - "Name" -> name
    - "Main picture" -> image_url

    Dataset structure:
    /market/merchant/product_id/
        - snapshot.wtl (or similar)
        - page.mhtml
        - screenshot.png
        - metadata.json (optional)
    """

    KLARNA_LABELS = {
        "Price": "price",
        "Name": "name",
        "Main picture": "image_url",
    }

    # Currency symbols to currency codes
    CURRENCY_SYMBOLS = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "kr": "SEK",
        "SEK": "SEK",
        "NOK": "NOK",
        "DKK": "DKK",
    }

    # Price parsing patterns
    PRICE_PATTERNS = [
        r"[\$€£¥][\s]*([0-9,]+\.?[0-9]*)",  # $99.99, € 99.99
        r"([0-9,]+\.?[0-9]*)[\s]*[\$€£¥]",  # 99.99$, 99.99 €
        r"([0-9]+)[,.]([0-9]{2})[\s]*(kr|SEK|NOK|DKK|EUR|USD|GBP)",  # 99,99 kr
        r"([0-9,]+)[\s]*(kr|SEK|NOK|DKK)",  # 99 kr
        r"([0-9]+\.[0-9]{2})",  # 99.99 (fallback)
    ]

    def parse_directory(self, dir_path: str) -> ParsedProduct:
        """
        Parse a product directory.

        Tries multiple file formats in order of preference.

        Args:
            dir_path: Path to product directory

        Returns:
            ParsedProduct with extracted data
        """
        dir_path = Path(dir_path)
        product = ParsedProduct(file_path=str(dir_path))

        # Try to extract market/merchant from path
        parts = dir_path.parts
        if len(parts) >= 2:
            product.market = parts[-3] if len(parts) >= 3 else None
            product.merchant = parts[-2]

        # Try different file formats
        mhtml_files = list(dir_path.glob("*.mhtml")) + list(dir_path.glob("*.mht"))
        wtl_files = list(dir_path.glob("*.wtl"))
        json_files = list(dir_path.glob("metadata.json"))

        # Try metadata.json first (if available)
        if json_files:
            self._parse_metadata_json(json_files[0], product)

        # Try MHTML
        if mhtml_files and not product.name:
            self._parse_mhtml(mhtml_files[0], product)

        # Try WTL
        if wtl_files and not product.name:
            self._parse_wtl(wtl_files[0], product)

        return product

    def parse_mhtml(self, file_path: str) -> ParsedProduct:
        """
        Parse an MHTML file.

        Args:
            file_path: Path to MHTML file

        Returns:
            ParsedProduct with extracted data
        """
        product = ParsedProduct(file_path=file_path)
        self._parse_mhtml(Path(file_path), product)
        return product

    def _parse_mhtml(self, file_path: Path, product: ParsedProduct) -> None:
        """Parse MHTML file and extract product data."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # Parse as email message
            msg = message_from_bytes(content)

            # Extract HTML content
            html_content = self._extract_html_from_mhtml(msg)
            if not html_content:
                product.errors.append("No HTML content found in MHTML")
                return

            # Parse HTML
            soup = BeautifulSoup(html_content, "lxml")
            self._extract_from_soup(soup, product)

            # Try to get source URL from MHTML headers
            if not product.source_url:
                product.source_url = self._extract_url_from_mhtml(msg)

        except Exception as e:
            product.errors.append(f"MHTML parse error: {str(e)}")

    def _extract_html_from_mhtml(self, msg) -> Optional[str]:
        """Extract HTML content from MHTML message."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")
        else:
            content_type = msg.get_content_type()
            if content_type == "text/html":
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")

        return None

    def _extract_url_from_mhtml(self, msg) -> Optional[str]:
        """Extract source URL from MHTML headers."""
        # Try Content-Location header
        if msg.is_multipart():
            for part in msg.walk():
                location = part.get("Content-Location")
                if location and location.startswith("http"):
                    return location
        return msg.get("Content-Location")

    def _parse_wtl(self, file_path: Path, product: ParsedProduct) -> None:
        """Parse WTL snapshot file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # WTL files can be JSON or HTML-like
            if content.strip().startswith("{"):
                self._parse_wtl_json(content, product)
            else:
                # Try parsing as HTML
                soup = BeautifulSoup(content, "lxml")
                self._extract_from_soup(soup, product)

        except Exception as e:
            product.errors.append(f"WTL parse error: {str(e)}")

    def _parse_wtl_json(self, content: str, product: ParsedProduct) -> None:
        """Parse WTL JSON format."""
        try:
            data = json.loads(content)
            self._traverse_wtl_dom(data, product)
        except json.JSONDecodeError as e:
            product.errors.append(f"WTL JSON parse error: {str(e)}")

    def _traverse_wtl_dom(self, node: Any, product: ParsedProduct) -> None:
        """Traverse WTL DOM structure to find labeled elements."""
        if isinstance(node, dict):
            # Check for klarna-ai-label attribute
            attrs = node.get("attributes", {})
            label = attrs.get("klarna-ai-label")

            if label in self.KLARNA_LABELS:
                field_name = self.KLARNA_LABELS[label]
                self._extract_wtl_value(node, field_name, product)

            # Recursively traverse children
            for child in node.get("children", []):
                self._traverse_wtl_dom(child, product)

        elif isinstance(node, list):
            for item in node:
                self._traverse_wtl_dom(item, product)

    def _extract_wtl_value(
        self,
        node: dict,
        field_name: str,
        product: ParsedProduct
    ) -> None:
        """Extract value from WTL node based on field type."""
        if field_name == "name":
            text = node.get("textContent", "").strip()
            if text:
                product.name = text

        elif field_name == "price":
            text = node.get("textContent", "").strip()
            if text:
                product.raw_price = text
                parsed = self._parse_price(text)
                if parsed:
                    product.price, product.currency = parsed

        elif field_name == "image_url":
            attrs = node.get("attributes", {})
            url = attrs.get("src") or attrs.get("data-src")
            if url:
                product.image_url = url

    def _extract_from_soup(self, soup: BeautifulSoup, product: ParsedProduct) -> None:
        """Extract product data from BeautifulSoup parsed HTML."""
        # Find elements with klarna-ai-label attribute
        for label, field_name in self.KLARNA_LABELS.items():
            elements = soup.find_all(attrs={"klarna-ai-label": label})

            for element in elements:
                if field_name == "name" and not product.name:
                    text = element.get_text(strip=True)
                    if text:
                        product.name = text
                        break

                elif field_name == "price" and product.price is None:
                    text = element.get_text(strip=True)
                    if text:
                        product.raw_price = text
                        parsed = self._parse_price(text)
                        if parsed:
                            product.price, product.currency = parsed
                            break

                elif field_name == "image_url" and not product.image_url:
                    # Try different image attributes
                    url = (
                        element.get("src") or
                        element.get("data-src") or
                        element.get("data-lazy-src")
                    )
                    if not url and element.name == "img":
                        url = element.get("src")
                    if not url:
                        # Check for img child
                        img = element.find("img")
                        if img:
                            url = img.get("src") or img.get("data-src")
                    if url:
                        product.image_url = url
                        break

        # Fallback: Try common selectors if klarna labels not found
        if not product.name:
            self._fallback_name_extraction(soup, product)

        if product.price is None:
            self._fallback_price_extraction(soup, product)

    def _fallback_name_extraction(
        self,
        soup: BeautifulSoup,
        product: ParsedProduct
    ) -> None:
        """Fallback name extraction using common selectors."""
        selectors = [
            'h1[itemprop="name"]',
            'h1.product-title',
            'h1.product-name',
            '[data-testid="product-title"]',
            '.product-title h1',
            'h1',
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) > 3:
                    product.name = text
                    break

    def _fallback_price_extraction(
        self,
        soup: BeautifulSoup,
        product: ParsedProduct
    ) -> None:
        """Fallback price extraction using common selectors."""
        selectors = [
            '[itemprop="price"]',
            '.product-price',
            '.price-current',
            '[data-testid="product-price"]',
            '.price',
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    parsed = self._parse_price(text)
                    if parsed:
                        product.price, product.currency = parsed
                        product.raw_price = text
                        break

    def _parse_metadata_json(
        self,
        file_path: Path,
        product: ParsedProduct
    ) -> None:
        """Parse metadata.json file if present."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("name"):
                product.name = data["name"]
            if data.get("price"):
                parsed = self._parse_price(str(data["price"]))
                if parsed:
                    product.price, product.currency = parsed
            if data.get("image_url"):
                product.image_url = data["image_url"]
            if data.get("source_url"):
                product.source_url = data["source_url"]
            if data.get("merchant"):
                product.merchant = data["merchant"]
            if data.get("market"):
                product.market = data["market"]

        except Exception as e:
            product.errors.append(f"Metadata JSON error: {str(e)}")

    def _parse_price(self, price_text: str) -> Optional[tuple[float, str]]:
        """
        Parse price string into numeric value and currency.

        Args:
            price_text: Raw price string

        Returns:
            Tuple of (price, currency) or None if parsing fails
        """
        if not price_text:
            return None

        # Clean up text
        text = price_text.strip()

        # Detect currency
        currency = "USD"
        for symbol, code in self.CURRENCY_SYMBOLS.items():
            if symbol in text:
                currency = code
                break

        # Try each pattern
        for pattern in self.PRICE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    # Get the numeric groups
                    groups = match.groups()
                    if len(groups) >= 2 and groups[1] and len(groups[1]) == 2:
                        # Format: 99,99 or 99.99
                        price_str = f"{groups[0]}.{groups[1]}"
                    else:
                        price_str = groups[0]

                    # Clean and parse
                    price_str = price_str.replace(",", "").replace(" ", "")
                    price = float(price_str)

                    if 0 < price < 1_000_000:  # Sanity check
                        return (price, currency)

                except (ValueError, IndexError):
                    continue

        return None

    def is_valid_product(self, product: ParsedProduct) -> bool:
        """Check if parsed product has minimum required data."""
        return bool(product.name and len(product.name) > 2)

    def to_dict(self, product: ParsedProduct) -> dict:
        """Convert ParsedProduct to dictionary for database storage."""
        return {
            "name": product.name,
            "price": product.price,
            "currency": product.currency,
            "image_url": product.image_url,
            "source_url": product.source_url,
            "merchant": product.merchant,
            "market": product.market,
        }
