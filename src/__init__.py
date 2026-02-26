"""
I-JEPA-Inspired Cross-Domain Reasoning System.

A system that reasons in latent embedding space using Joint Embedding
Predictive Architecture (JEPA), then decodes to human-readable output.
"""

from .encoders import (
    TransformerEncoder,
    CodeContextEncoder,
    MathContextEncoder,
    TextContextEncoder,
    TargetEncoder,
    build_encoders,
)
from .predictors import LatentPredictor, MLPPredictor, build_predictor
from .decoders import LatentConditionedDecoder, PretrainedDecoder, build_decoder
from .models import JEPAReasoner, MultiDomainJEPAReasoner
from .llm_interface import Verifier, build_llm_interface
from .latent_graph import LatentGraphSearch, LatentRefiner, LatentDenoiser

__all__ = [
    "JEPAReasoner",
    "MultiDomainJEPAReasoner",
    "TransformerEncoder",
    "CodeContextEncoder",
    "MathContextEncoder",
    "TextContextEncoder",
    "TargetEncoder",
    "build_encoders",
    "LatentPredictor",
    "MLPPredictor",
    "build_predictor",
    "LatentConditionedDecoder",
    "PretrainedDecoder",
    "build_decoder",
    "Verifier",
    "build_llm_interface",
    "LatentGraphSearch",
    "LatentRefiner",
    "LatentDenoiser",
]
