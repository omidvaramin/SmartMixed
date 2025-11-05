# SmartMixed: Adaptive Activation Function Neural Network for MNIST

This repository implements **SmartMixed**, a novel two-phase training strategy for adaptive activation function learning in neural networks. The approach combines adaptive activation selection with a sophisticated training methodology for MNIST classification, where each neuron can learn its optimal activation function from a pool (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, SELU) using a Gumbel-Softmax mechanism.

## Features
- **SmartMixed Training Strategy:** Novel two-phase training approach (Selective → Mixed) for optimal activation function learning
- **Multiple Model Types:** Support for custom, selective, mixed, and smartmixed architectures
- **Comprehensive Experimentation:** Iterate mode for running experiments across all architectures and model types
- **Adaptive Activation Selection:** Each neuron in adaptive layers selects its activation function from a pool, learned via Gumbel-Softmax
- **Activation Pool:** Supports ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, and SELU
- **Rich Visualization:** Saves per-neuron activation choices, bar charts, and per-epoch activation trends
- **Robust Inference Pipeline:** Reconstructs networks from saved results and evaluates on test sets
- **Device Support:** Runs on CPU, CUDA, or MPS (Apple Silicon) automatically
- **Reproducibility:** Fixed random seed and deterministic data splits

## Quick Start

### Running Complete Experiments (Recommended)
To run experiments across all architectures and model types (default behavior):
```sh
python main.py
```
or explicitly:
```sh
python main.py --model iterate
```

This will execute comprehensive experiments using the **iterate mode**, which is the default and recommended approach for complete model evaluation.

### Training Individual Models
```sh
# Train SmartMixed model (two-phase training)
python main.py --model smartmixed --epochs 400 --transition_epoch 50

# Train selective model only
python main.py --model selective --epochs 200

# Train custom fixed-activation model
python main.py --model custom --epochs 200
```

### Inference
```sh
python main.py --inference_folder <output_folder>
```

## Command-Line Parameters
- `--model {custom,selective,mixed,smartmixed,iterate}`: Model type (default: `iterate`)
  - `custom`: Fixed activation functions
  - `selective`: Adaptive activation selection using Gumbel-Softmax
  - `mixed`: Fixed network created from a trained selective model
  - `smartmixed`: Two-phase training (selective → mixed)
  - `iterate`: Run experiments across all architectures and specified model types
- `--epochs N`: Number of training epochs (default: 400)
- `--lr LR`: Learning rate (default: 1e-4)
- `--batch_size N`: Batch size (default: 128)
- `--transition_epoch N`: Epoch to transition from selective to mixed in smartmixed mode (default: 50)
- `--iterate_choice STR`: Comma-separated model types for iterate mode (default: "custom,smartmixed")
- `--architectures STR`: Comma-separated architecture IDs for iterate mode (default: all architectures)
- `--small_size`: Use reduced dataset for quick debugging runs
- `--verbose`: Print detailed training statistics
- `--inference_folder PATH`: Run inference using a trained model
- `--gumbel_hard/--no-gumbel_hard`: Use hard (one-hot) or soft Gumbel-Softmax (default: hard)
- `--list_architectures`: Show all available architectures
- `--show_mnist_info`: Display MNIST dataset information

## Analysis Tools

### Performance Aggregator (`sandbox/aggregator.py`)
Comprehensive analysis tool that aggregates results from completed experiments:
- **Excel Reports:** Generates detailed performance analysis with multiple sheets
- **Ranking Analysis:** Creates first-rank frequency charts and ranking distributions
- **Statistical Metrics:** Computes Mean Reciprocal Rank (MRR) for activation function performance
- **Visualization:** Generates publication-ready charts and graphs
- **Architecture Comparison:** Per-architecture ranking tables and summaries

Usage:
```sh
cd sandbox
python aggregator.py
```

### Weight Analysis (`sandbox/weight_finder.py`)
Advanced tool for analyzing neural network weight patterns and activation function relationships:
- **Weight Extraction:** Extracts weight matrices from trained SmartMixed models
- **Activation Mapping:** Maps activation functions for each neuron
- **Connection Analysis:** Analyzes connections between neurons with different activation functions
- **Pattern Discovery:** Identifies weight distribution patterns across activation types

Usage:
```sh
cd sandbox
python weight_finder.py
```

## Output Structure
- Results are saved in `outputs/<experiment_type>_<timestamp>/`
- **Individual Models:** Architecture info, model weights, activation choices, visualizations, metrics
- **Experiment Iterations:** Organized by architecture with comprehensive summaries
- **Analysis Results:** Charts, Excel reports, and statistical summaries
- **Inference Results:** Saved in `inference/` subfolders

## Installation
```sh
pip install -r requirements.txt
```

## Requirements
- Python 3.9+
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- numpy, pandas, matplotlib, seaborn
- scikit-learn, openpyxl

## Examples

### Complete Experimental Pipeline
```sh
# Run full experiments (recommended)
python main.py --model iterate --epochs 400

# Run specific architectures only
python main.py --model iterate --architectures "1,5,10" --epochs 200

# Run specific model types
python main.py --model iterate --iterate_choice "custom,smartmixed" --epochs 300
```

### SmartMixed Training
```sh
# Two-phase training with custom transition point
python main.py --model smartmixed --epochs 400 --transition_epoch 100 --lr 1e-4
```

### Analysis and Visualization
```sh
# Generate comprehensive analysis after experiments
cd sandbox
python aggregator.py

# Analyze weight patterns
python weight_finder.py
```

## Citation

This work introduces the SmartMixed training strategy for adaptive activation function learning. For detailed methodology and results, please refer to:

```bibtex
@article{omidvar2025smartmixed,
  title={SmartMixed: A Two-Phase Training Strategy for Adaptive Activation Function Learning in Neural Networks},
  author={Omidvar, Amin},
  journal={arXiv preprint arXiv:2510.22450},
  year={2025}
}
```

**Paper Link:** https://arxiv.org/abs/2510.22450

---

For detailed implementation details, see the code and comments in `main.py`, `models.py`, `train.py`, and the analysis tools in the `sandbox/` directory.
