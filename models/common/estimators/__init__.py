"""Forecast models that are not plain scikit-learn objects, one module per model."""
from .persistence_escalation import PersistencePlusEscalation

__all__ = ["PersistencePlusEscalation"]
