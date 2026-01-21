"""
Encryption service for sensitive data protection.
"""
import base64
import hashlib
import json
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.config.settings import get_settings


class EncryptionService:
    """
    Service for encrypting and decrypting sensitive data.

    Uses Fernet symmetric encryption with PBKDF2 key derivation.
    """

    # Salt for key derivation (in production, use unique per-deployment)
    DEFAULT_SALT = b"price_compare_v1"

    def __init__(self, encryption_key: Optional[str] = None):
        settings = get_settings()
        key = encryption_key or settings.encryption_key.get_secret_value()

        if not key:
            # Generate a random key if none provided
            key = Fernet.generate_key().decode()

        self.fernet = self._derive_key(key)

    def _derive_key(self, password: str) -> Fernet:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Password/key string

        Returns:
            Fernet instance with derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.DEFAULT_SALT,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """
        Encrypt a string.

        Args:
            data: Plain text string

        Returns:
            Encrypted string (base64 encoded)
        """
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_data: Encrypted string (base64 encoded)

        Returns:
            Decrypted plain text string

        Raises:
            InvalidToken: If decryption fails
        """
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_dict(self, data: dict, fields: Optional[list[str]] = None) -> dict:
        """
        Encrypt specific fields in a dictionary.

        Args:
            data: Dictionary to encrypt
            fields: Fields to encrypt (if None, encrypts entire dict as JSON)

        Returns:
            Dictionary with encrypted fields
        """
        result = data.copy()

        if fields is None:
            # Encrypt entire dict as JSON
            json_str = json.dumps(data)
            return {"_encrypted": self.encrypt(json_str)}

        for field in fields:
            if field in result and result[field] is not None:
                value = str(result[field])
                result[field] = self.encrypt(value)

        return result

    def decrypt_dict(self, data: dict, fields: Optional[list[str]] = None) -> dict:
        """
        Decrypt specific fields in a dictionary.

        Args:
            data: Dictionary with encrypted fields
            fields: Fields to decrypt (if None and _encrypted exists, decrypts entire dict)

        Returns:
            Dictionary with decrypted fields
        """
        if "_encrypted" in data and fields is None:
            # Decrypt entire dict
            json_str = self.decrypt(data["_encrypted"])
            return json.loads(json_str)

        result = data.copy()
        fields = fields or []

        for field in fields:
            if field in result and result[field] is not None:
                try:
                    result[field] = self.decrypt(result[field])
                except InvalidToken:
                    # Field might not be encrypted
                    pass

        return result

    def encrypt_json(self, data: Any) -> str:
        """
        Encrypt any JSON-serializable data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted string
        """
        json_str = json.dumps(data)
        return self.encrypt(json_str)

    def decrypt_json(self, encrypted_data: str) -> Any:
        """
        Decrypt JSON data.

        Args:
            encrypted_data: Encrypted JSON string

        Returns:
            Decrypted data
        """
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)

    @staticmethod
    def hash_query(query: str, query_type: str = "text") -> str:
        """
        Generate a hash for a search query (for caching).

        Args:
            query: Search query string
            query_type: Type of query (text, url, image)

        Returns:
            SHA256 hash of the query
        """
        combined = f"{query_type}:{query}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def hash_file(file_content: bytes) -> str:
        """
        Generate a hash for file content.

        Args:
            file_content: File bytes

        Returns:
            SHA256 hash of the file
        """
        return hashlib.sha256(file_content).hexdigest()

    def is_encrypted(self, data: str) -> bool:
        """
        Check if a string appears to be encrypted.

        Args:
            data: String to check

        Returns:
            True if string appears to be Fernet-encrypted
        """
        try:
            # Fernet tokens start with gAAAAA
            if not data.startswith("gAAAAA"):
                return False
            # Try to decode as base64
            base64.urlsafe_b64decode(data)
            return True
        except Exception:
            return False


def generate_encryption_key() -> str:
    """
    Generate a new random encryption key.

    Returns:
        Base64-encoded encryption key suitable for .env file
    """
    return Fernet.generate_key().decode()
