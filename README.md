# 🧠 NeuroBridge

> A neural interface bridge system for processing and analyzing brain-computer interface (BCI) data. Decodes intracranial electrocorticography (ECoG) signals into phonemes using deep learning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

NeuroBridge replicates the core architecture of a high-performance speech neuroprosthesis, inspired by groundbreaking research at Stanford and UC Davis. The system translates intracranial ECoG signals directly into intelligible speech via phoneme decoding.

### Architecture

- **Offline Decoder**: Bidirectional LSTM for batch analysis (2.4M parameters)
- **Real-time Decoder**: Unidirectional LSTM for causal inference
- **REST API**: Flask-based server for signal processing
- **Speech Synthesis**: Placeholder vocoder integration

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/yourusername/neurobridge.git
cd neurobridge
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env to customize settings
```

### Usage

```bash
# Show system info
python neurobridge.py info

# Train the model (with mock data)
python neurobridge.py train --epochs 5

# Evaluate the trained model
python neurobridge.py evaluate --samples 100

# Run a single prediction
python neurobridge.py predict --top-k 5

# Start the REST API server
python neurobridge.py serve --port 5000
```

## API

Once the server is running, the following endpoints are available:

| Method | Endpoint       | Description                          |
|--------|----------------|--------------------------------------|
| GET    | `/api/health`  | Health check                         |
| POST   | `/api/process` | Process ECoG signal, get phonemes    |
| POST   | `/api/analyze` | Analyze neural data                  |

See [API.md](./API.md) for full documentation.

### Example: Process Signal

```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ...], "sampling_rate": 1000}'
```

## Project Structure

```
neurobridge/
├── __init__.py        # Package exports
├── config.py          # Environment-based configuration
├── model.py           # LSTM decoder architectures
├── data.py            # Data loading & preprocessing
├── train.py           # Training pipeline with callbacks
├── evaluate.py        # Evaluation metrics
├── inference.py       # Real-time inference
├── synthesizer.py     # Speech synthesis placeholder
└── api.py             # Flask REST API
```

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Docker

```bash
docker build -t neurobridge .
docker run -p 5000:5000 neurobridge
```

## Environment Variables

All configuration can be overridden via environment variables. See [.env.example](./.env.example) for the full list.

| Variable               | Default | Description                    |
|------------------------|---------|--------------------------------|
| `NB_NUM_TIMESTEPS`     | 100     | Sequence length per sample     |
| `NB_NUM_FEATURES`      | 128     | Number of ECoG channels        |
| `NB_NUM_PHONEMES`      | 41      | Phoneme vocabulary size        |
| `NB_BATCH_SIZE`        | 32      | Training batch size            |
| `NB_EPOCHS`            | 5       | Training epochs                |
| `NB_API_PORT`          | 5000    | API server port                |

## References

This project is inspired by:

1. Stanford speech neuroprosthesis research
2. UC Davis BCI phoneme decoding work
3. Modern deep learning approaches to neural signal processing

## License

MIT License — see [LICENSE](./LICENSE) for details.
