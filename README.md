# ETHOS Transformer for EHR Data

This repository implements an ETHOS-like transformer model for Electronic Health Record (EHR) data, based on the paper "Zero shot health trajectory prediction using transformer" by Renc et al. The implementation provides a complete pipeline for processing OMOP and MEDS format EHR data, training a transformer model, and performing zero-shot inference.

## Overview

ETHOS (Enhanced Transformer for Health Outcome Simulation) is a novel application of transformer architecture for analyzing high-dimensional, heterogeneous, and episodic health data. The model processes Patient Health Timelines (PHTs) - detailed, tokenized records of health events - to predict future health trajectories using zero-shot learning.

## Features

- **Data Processing**: Convert OMOP and MEDS format EHR data to tokenized Patient Health Timelines
- **Transformer Model**: Implementation of ETHOS architecture with learnable positional encodings
- **Training Pipeline**: Complete training script with validation, checkpointing, and visualization
- **Zero-shot Inference**: Predict mortality, readmission, SOFA scores, and length of stay without task-specific training
- **Timeline Generation**: Generate future patient health trajectories
- **Comprehensive Analysis**: Analyze patient timelines and generate insights

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cursor_transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p omop_data meds_data processed_data models logs plots
```

## Data Preparation

### OMOP Format
Place your OMOP data in the `omop_data/` directory with the following structure:
```
omop_data/
├── person/
│   ├── part_0.parquet
│   ├── part_1.parquet
│   └── ...
├── visit_occurrence/
│   ├── part_0.parquet
│   └── ...
├── condition_occurrence/
├── drug_exposure/
├── procedure_occurrence/
├── measurement/
├── observation/
└── death/
```

### MEDS Format
Place your MEDS data in the `meds_data/` directory as parquet files containing event data with columns like `patient_id`, `timestamp`, `event_type`, etc.

## Usage

### 1. Data Processing

First, process your EHR data to create tokenized Patient Health Timelines:

```bash
python data_processor.py
```

This will:
- Load OMOP and MEDS data
- Create patient timelines
- Tokenize events and measurements
- Build vocabulary
- Save processed data to `processed_data/`

### 2. Training

Train the ETHOS transformer model:

```bash
python train.py --batch_size 32 --max_epochs 100 --learning_rate 3e-4
```

Training options:
- `--batch_size`: Training batch size (default: 32)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--device`: Device to use (auto/cuda/cpu, default: auto)
- `--resume`: Resume from checkpoint

### 3. Inference

Run inference with the trained model:

```bash
python inference.py --model_path models/best_checkpoint.pth --patient_id 12345
```

Inference options:
- `--model_path`: Path to trained model checkpoint
- `--patient_id`: Specific patient ID to analyze (optional)
- `--output_dir`: Directory for inference results (default: inference_results)

## Model Architecture

The ETHOS transformer implements:

- **Decoder-only architecture** with causal masking
- **Learnable positional encodings** instead of fixed sinusoidal
- **Multi-head self-attention** with configurable dimensions
- **Feed-forward networks** with residual connections
- **Layer normalization** and dropout for regularization

### Configuration

Model parameters can be adjusted in `config.py`:

```python
@dataclass
class ModelConfig:
    d_model: int = 768          # Model dimension
    n_heads: int = 12           # Number of attention heads
    n_layers: int = 12          # Number of transformer layers
    d_ff: int = 3072           # Feed-forward dimension
    max_seq_len: int = 2048    # Maximum sequence length
    dropout: float = 0.1       # Dropout rate
```

## Tokenization Strategy

The implementation uses a sophisticated tokenization approach:

1. **Event Type Tokens**: ADM (admission), DIS (discharge), COND (condition), etc.
2. **Concept Tokens**: Specific medical concepts (ICD codes, ATC codes, etc.)
3. **Quantile Tokens**: Numerical values converted to quantiles (Q1-Q10)
4. **Time Interval Tokens**: Temporal gaps between events (5m, 15m, 1h, 1d, etc.)
5. **Static Tokens**: Patient demographics, age intervals, birth year

## Training Process

The training follows the ETHOS methodology:

1. **Data Preparation**: Convert EHR data to chronological patient timelines
2. **Tokenization**: Transform events into token sequences
3. **Sequence Modeling**: Train transformer to predict next tokens
4. **Zero-shot Learning**: Model learns to generate future health trajectories

## Inference Capabilities

The trained model can perform zero-shot predictions:

- **Mortality Prediction**: Estimate patient mortality probability
- **Readmission Risk**: Predict readmission within specified timeframes
- **SOFA Score Estimation**: Predict Sequential Organ Failure Assessment scores
- **Length of Stay**: Estimate hospital/ICU length of stay
- **Timeline Generation**: Generate future patient health trajectories

## Example Workflow

```python
from data_processor import EHRDataProcessor
from transformer_model import create_ethos_model
from inference import ETHOSInference

# 1. Process data
processor = EHRDataProcessor()
tokenized_timelines, vocab = processor.process_all_data()

# 2. Create and train model
model = create_ethos_model(len(vocab))
# ... training code ...

# 3. Run inference
inference = ETHOSInference('models/best_checkpoint.pth', 'processed_data/vocabulary.pkl')
analysis = inference.analyze_patient_timeline(patient_timeline)
future_timeline = inference.generate_future_timeline(patient_timeline)
```

## Output Files

The pipeline generates several output files:

- `processed_data/`: Tokenized timelines, vocabulary, and mappings
- `models/`: Model checkpoints and weights
- `logs/`: Training logs and metrics
- `plots/`: Training curves and visualizations
- `inference_results/`: Inference results and timeline visualizations

## Performance Considerations

- **Memory**: Large models may require significant GPU memory
- **Batch Size**: Adjust based on available memory
- **Sequence Length**: Longer sequences require more memory and computation
- **Data Size**: Larger datasets improve model performance but increase training time

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Data Loading Errors**: Check file paths and parquet file integrity
3. **Training Divergence**: Reduce learning rate or increase gradient clipping
4. **Slow Training**: Use GPU acceleration and optimize data loading

### Performance Tips

- Use SSD storage for faster data loading
- Enable mixed precision training for faster GPU training
- Use multiple workers for data loading
- Monitor GPU memory usage during training

## Citation

If you use this implementation, please cite the original ETHOS paper:

```bibtex
@article{renc2024zero,
  title={Zero shot health trajectory prediction using transformer},
  author={Renc, Pawel and Jia, Yugang and Samir, Anthony E and others},
  journal={npj Digital Medicine},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on the repository or contact the maintainers.

## Acknowledgments

This implementation is based on the ETHOS paper and builds upon the transformer architecture introduced in "Attention Is All You Need" by Vaswani et al.
