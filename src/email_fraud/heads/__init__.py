"""Heads subpackage — import triggers @register side-effects."""

from email_fraud.heads.base import BaseHead
from email_fraud.heads.prototypical import PrototypicalHead

__all__ = ["BaseHead", "PrototypicalHead"]
