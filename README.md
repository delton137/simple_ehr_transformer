# ETHOS Transformer for EHR Data

This repository implements an ETHOS-like transformer model for Electronic Health Record (EHR) data, based on the paper "Zero shot health trajectory prediction using transformer" by Renc et al. The implementation provides a complete pipeline for processing OMOP format EHR data, training a transformer model, and performing zero-shot inference.

## Overview

ETHOS (Enhanced Transformer for Health Outcome Simulation) is a novel application of transformer architecture for analyzing high-dimensional, heterogeneous, and episodic health data. The model processes Patient Health Timelines (PHTs) - detailed, tokenized records of health events - to predict future health trajectories using zero-shot learning.

## Features

- **Data Processing**: Convert OMOP format EHR data to tokenized Patient Health Timelines
- **Large Dataset Optimization**: Memory management and chunked processing for datasets several GB in size
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
mkdir -p processed_data models logs plots
```

## Data Preparation

### OMOP Format
Place your OMOP data in a directory with the following structure:
```
your_omop_data/
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

**Note**: The code expects parquet files organized in subdirectories by table name. Each table subdirectory should contain one or more parquet files.

## Usage

### 1. Data Processing

First, process your OMOP data to create tokenized Patient Health Timelines:

```bash
# Use default path (omop_data/)
python data_processor.py

# Specify custom OMOP data path
python data_processor.py --data_path /path/to/your/omop_data

# Use dataset tag for isolation (recommended for multiple datasets)
python data_processor.py --data_path /path/to/omop_data --tag aou_2023

# Specify custom output directory
python data_processor.py --data_path /path/to/omop_data --output_dir /path/to/output

# Adjust memory limit for large datasets
python data_processor.py --data_path /path/to/omop_data --memory_limit 16.0

# Force reprocessing (useful for debugging)
python data_processor.py --data_path /path/to/omop_data --tag aou_2023 --force_reprocess
```

**Command line options:**
- `--data_path`: Path to OMOP data directory (default: `omop_data/`)
- `--tag`: Dataset tag for isolating different datasets (e.g., `aou_2023`, `mimic_iv`, `eicu`)
- `--output_dir`: Output directory for processed data (default: `processed_data/` or `processed_data_{tag}/`)
- `--memory_limit`: Memory limit in GB (default: 8.0)
- `--force_reprocess`: Force reprocessing even if data exists

**Dataset Isolation with Tags:**
The tag system allows you to work with multiple datasets simultaneously:

```bash
# Process All of Us 2023 data
python data_processor.py --data_path ~/omop_data_2023 --tag aou_2023

# Process MIMIC-IV data
python data_processor.py --data_path ~/mimic_iv --tag mimic_iv

# Process eICU data
python data_processor.py --data_path ~/eicu --tag eicu
```

This creates separate directories:
- `processed_data_aou_2023/` - All of Us 2023 processed data
- `processed_data_mimic_iv/` - MIMIC-IV processed data  
- `processed_data_eicu/` - eICU processed data

### 2. Training

Train the ETHOS transformer model:

```bash
# Train with default data directory
python train.py --batch_size 32 --max_epochs 100 --learning_rate 3e-4

# Train with tagged dataset
python train.py --tag aou_2023 --batch_size 32 --max_epochs 100

# Train with custom data directory
python train.py --data_dir processed_data_aou_2023 --batch_size 32
```

Training options:
- `--tag`: Dataset tag to use (automatically finds `processed_data_{tag}/`)
- `--data_dir`: Directory containing processed data (default: `processed_data/`)
- `--batch_size`: Training batch size (default: 32)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--device`: Device to use (auto/cuda/cpu, default: auto)
- `--resume`: Resume from checkpoint

**Tag-based Training:**
```bash
# Train on All of Us 2023 data
python train.py --tag aou_2023 --batch_size 64 --max_epochs 200

# Train on MIMIC-IV data  
python train.py --tag mimic_iv --batch_size 32 --max_epochs 100

# Models are saved to separate directories:
# - models_aou_2023/
# - models_mimic_iv/
```

### 3. Inference

Run inference with the trained model:

```bash
# Basic inference
python inference.py --model_path models/best_checkpoint.pth --patient_id 12345

# Inference with tagged dataset
python inference.py --tag aou_2023 --model_path models_aou_2023/best_checkpoint.pth

# Inference with custom data directory
python inference.py --model_path models/best_checkpoint.pth --data_dir processed_data_aou_2023
```

Inference options:
- `--tag`: Dataset tag to use (automatically finds `processed_data_{tag}/`)
- `--model_path`: Path to trained model checkpoint
- `--data_dir`: Directory containing processed data (default: `processed_data/`)
- `--patient_id`: Specific patient ID to analyze (optional)
- `--output_dir`: Directory for inference results (default: `inference_results/` or `inference_results_{tag}/`)

**Tag-based Inference:**
```bash
# Analyze All of Us 2023 patients
python inference.py --tag aou_2023 --model_path models_aou_2023/best_checkpoint.pth

# Analyze MIMIC-IV patients
python inference.py --tag mimic_iv --model_path models_mimic_iv/best_checkpoint.pth

# Results are saved to separate directories:
# - inference_results_aou_2023/
# - inference_results_mimic_iv/
```

### 4. Example Workflow

Run the complete workflow example:

```bash
# Basic workflow
python example_workflow.py --data_path ~/omop_data

# Workflow with dataset tag
python example_workflow.py --data_path ~/omop_data_2023 --tag aou_2023

# Workflow with custom memory limit
python example_workflow.py --data_path ~/omop_data --tag aou_2023 --memory_limit 16.0
```

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

@dataclass
class DataConfig:
    chunk_size: int = 10000     # Process data in chunks
    max_patients_per_chunk: int = 5000  # Max patients in memory
    memory_limit_gb: float = 8.0        # Memory limit for processing
```

## Tokenization Strategy

The implementation uses a sophisticated tokenization approach:

1. **Event Type Tokens**: ADM (admission), DIS (discharge), COND (condition), etc.
2. **Concept Tokens**: Specific medical concepts (ICD codes, ATC codes, etc.)
3. **Quantile Tokens**: Numerical values converted to quantiles (Q1-Q10)
4. **Time Interval Tokens**: Temporal gaps between events (5m, 15m, 1h, 1d, etc.)
5. **Static Tokens**: Patient demographics, age intervals, birth year

## Large Dataset Optimization

The code is specifically optimized for large OMOP datasets:

- **Chunked Processing**: Data is processed in manageable chunks to control memory usage
- **Memory Monitoring**: Real-time memory usage tracking with configurable limits
- **Garbage Collection**: Automatic memory cleanup between processing steps
- **Parallel Processing**: Support for multiprocessing when available
- **Streaming**: Processes parquet files without loading entire tables into memory

## Training Process

The training follows the ETHOS methodology:

1. **Data Preparation**: Convert OMOP data to chronological patient timelines
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
from data_processor import OMOPDataProcessor
from transformer_model import create_ethos_model
from inference import ETHOSInference

# 1. Process data with custom path
processor = OMOPDataProcessor(data_path="/path/to/omop_data")
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
- **Chunk Size**: Adjust chunk size based on available RAM

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, sequence length, or chunk size
2. **Data Loading Errors**: Check file paths and parquet file integrity
3. **Training Divergence**: Reduce learning rate or increase gradient clipping
4. **Slow Training**: Use GPU acceleration and optimize data loading

### Performance Tips

- Use SSD storage for faster data loading
- Enable mixed precision training for faster GPU training
- Use multiple workers for data loading
- Monitor GPU memory usage during training
- Adjust memory limits based on your system

### Large Dataset Tips

- Start with smaller chunks and increase gradually
- Monitor memory usage during processing
- Use `--memory_limit` to set appropriate limits for your system
- Process data on machines with sufficient RAM

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
