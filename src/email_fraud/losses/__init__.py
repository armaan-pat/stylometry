"""Loss subpackage — import triggers @register side-effects."""

from email_fraud.losses.base import BaseLoss
from email_fraud.losses.contrastive import ContrastiveLoss
from email_fraud.losses.episodic import EpisodicPrototypeLoss
from email_fraud.losses.supcon import SupConLoss
from email_fraud.losses.triplet import TripletLoss

__all__ = [
    "BaseLoss",
    "ContrastiveLoss",
    "EpisodicPrototypeLoss",
    "SupConLoss",
    "TripletLoss",
]
