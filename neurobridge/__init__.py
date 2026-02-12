"""
NeuroBridge - Neural Interface Bridge System

A clinical-grade neural interface bridge system for processing and analyzing
brain-computer interface (BCI) data. Decodes intracranial electrocorticography (ECoG) 
and high-density EEG signals into phonemes using Bidirectional LSTMs.
"""

__version__ = "1.0.0"
__author__ = "NeuroBridge Team"

# Core architecture imports for top-level package access
from neurobridge.config import Config
from neurobridge.data.ingestion import ECoGIngestionPipeline
from neurobridge.processing.signal import SignalProcessor
from neurobridge.model.decoder import NeurobridgeDecoder
from neurobridge.training.trainer import ModelTrainer
from neurobridge.actuation.interface import ProstheticInterface
