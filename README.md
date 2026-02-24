# Environmental Sound Classification

Production-ready environmental sound classification system using deep learning. This project implements a CRNN (Convolutional Recurrent Neural Network) model trained on the ESC-50 dataset for classifying 50 different environmental sounds.

## Features

- **Modern Architecture**: CRNN model with CNN feature extraction and RNN temporal modeling
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, confusion matrix, and top-K accuracy
- **Data Augmentation**: SpecAugment, MixUp, and other augmentation techniques
- **Interactive Demo**: Streamlit web application for real-time classification
- **Production Ready**: Proper logging, checkpointing, configuration management, and documentation
- **Privacy Focused**: Research-oriented with clear disclaimers and privacy considerations

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Environmental-Sound-Classification.git
cd Environmental-Sound-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

### Training

1. **With ESC-50 Dataset** (recommended):
```bash
python scripts/train.py
```

2. **With Synthetic Data** (for testing):
The system automatically falls back to synthetic data if ESC-50 is not available.

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   └── crnn.py              # CRNN model implementation
│   ├── data/                     # Dataset classes
│   │   └── esc50_dataset.py     # ESC-50 dataset loader
│   ├── features/                 # Feature extraction
│   │   ├── melspectrogram.py    # Mel-spectrogram extractor
│   │   └── augmentation.py      # Data augmentation
│   ├── losses/                   # Loss functions
│   │   └── focal_loss.py        # Focal loss and others
│   ├── metrics/                  # Evaluation metrics
│   │   └── classification.py   # Classification metrics
│   ├── train/                    # Training utilities
│   │   └── trainer.py          # Training loop
│   ├── eval/                     # Evaluation utilities
│   │   └── evaluator.py        # Evaluation and visualization
│   └── utils/                    # Utility functions
│       ├── device.py            # Device management
│       └── audio.py             # Audio processing
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model/                   # Model configurations
│   ├── data/                    # Data configurations
│   ├── training/                # Training configurations
│   └── evaluation/             # Evaluation configurations
├── scripts/                      # Training and evaluation scripts
│   └── train.py                # Main training script
├── demo/                         # Interactive demo
│   └── streamlit_app.py        # Streamlit web app
├── tests/                        # Unit tests
├── data/                         # Data directory
│   ├── raw/                     # Raw data (ESC-50)
│   └── processed/              # Processed data
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── assets/                       # Evaluation results and visualizations
└── requirements.txt              # Python dependencies
```

## Model Architecture

The CRNN model combines:

1. **CNN Feature Extractor**: 4-layer CNN with batch normalization and ReLU activation
2. **RNN Temporal Modeling**: 2-layer bidirectional LSTM
3. **Classification Head**: Fully connected layers with dropout

### Key Features:
- **Input**: Mel-spectrogram (128 bins × time frames)
- **Parameters**: ~2M trainable parameters
- **Architecture**: CNN → LSTM → FC layers
- **Regularization**: Dropout, batch normalization, gradient clipping

## Dataset

### ESC-50 Dataset
- **50 classes** of environmental sounds
- **2,000 audio files** (5 seconds each)
- **44.1 kHz** sample rate
- **Categories**: Animals, Nature, Human sounds, Machines, etc.

### Data Splits:
- **Training**: Folds 1-3 (1,200 samples)
- **Validation**: Fold 4 (400 samples)
- **Testing**: Fold 5 (400 samples)

### Data Augmentation:
- **SpecAugment**: Time and frequency masking
- **MixUp**: Mixing samples and labels
- **Time Stretching**: Temporal augmentation
- **Pitch Shifting**: Frequency augmentation

## Training

### Configuration
Training is configured via Hydra configuration files in `configs/`. Key parameters:

```yaml
# Model configuration
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true

# Training configuration
training:
  max_epochs: 100
  patience: 15
  batch_size: 32
  learning_rate: 0.001
```

### Training Process
1. **Data Loading**: ESC-50 dataset with augmentation
2. **Feature Extraction**: Mel-spectrogram (128 bins)
3. **Model Training**: CRNN with AdamW optimizer
4. **Validation**: Per-epoch validation with early stopping
5. **Checkpointing**: Save best model and regular checkpoints

### Monitoring
- **TensorBoard**: Real-time training metrics
- **Logging**: Structured logging with timestamps
- **Checkpoints**: Automatic saving of best models

## Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro, micro, and weighted F1
- **Precision/Recall**: Per-class and macro averages
- **Top-K Accuracy**: Top-5 accuracy for robustness
- **Confusion Matrix**: Detailed error analysis

### Visualizations
- **Confusion Matrix**: Normalized classification errors
- **Per-Class Accuracy**: Individual class performance
- **Confidence Distribution**: Prediction confidence analysis
- **Top-K Accuracy**: Robustness across different K values

### Results
Typical performance on ESC-50 test set:
- **Accuracy**: ~85-90%
- **Macro F1**: ~0.85-0.90
- **Top-5 Accuracy**: ~95-98%

## Demo Application

### Streamlit Web App
Interactive demo with:
- **Audio Upload**: Support for WAV, MP3, FLAC, M4A, OGG
- **Real-time Classification**: Instant prediction results
- **Visualization**: Mel-spectrogram and prediction plots
- **Top-K Results**: Detailed probability rankings

### Usage
1. **Upload Audio**: Use the sidebar to upload audio files
2. **View Results**: See predictions with confidence scores
3. **Analyze**: Examine spectrograms and detailed results

## Configuration

### Hydra Configuration
The project uses Hydra for configuration management:

```bash
# Override configuration
python scripts/train.py model.hidden_size=256 training.batch_size=64

# Use different model
python scripts/train.py model=transformer

# Change data configuration
python scripts/train.py data.batch_size=16 data.augmentation.enabled=false
```

### Key Configuration Files:
- `configs/config.yaml`: Main configuration
- `configs/model/crnn.yaml`: CRNN model settings
- `configs/data/esc50.yaml`: Dataset configuration
- `configs/training/default.yaml`: Training parameters

## Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality
- **Testing**: Pytest unit tests

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## Privacy and Ethics

### Important Disclaimers

**This is a research demonstration project. It is NOT intended for:**

- **Biometric Identification**: Do not use for identifying individuals
- **Production Deployment**: Not validated for critical applications
- **Voice Cloning**: Prohibited for creating synthetic voices
- **Surveillance**: Not designed for monitoring or tracking

### Privacy Considerations

- **No Data Storage**: Audio files are processed locally
- **No Personal Information**: No PII is logged or stored
- **Research Only**: Intended for educational and research purposes
- **Open Source**: Full transparency in implementation

### Ethical Guidelines

1. **Respect Privacy**: Only process audio you have permission to analyze
2. **Transparent Use**: Clearly communicate the system's capabilities and limitations
3. **No Misuse**: Do not use for unauthorized surveillance or identification
4. **Research Ethics**: Follow institutional guidelines for audio data research

## Performance

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB+ recommended
- **GPU**: Optional, CUDA/MPS support available
- **Storage**: 2GB+ for dataset and checkpoints

### Optimization
- **Mixed Precision**: Automatic mixed precision training
- **Gradient Accumulation**: Memory-efficient training
- **Data Loading**: Multi-process data loading
- **Device Fallback**: Automatic CUDA → MPS → CPU fallback

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient accumulation
   - Use CPU training

2. **Dataset Not Found**:
   - System automatically uses synthetic data
   - Download ESC-50 manually if needed

3. **Audio Loading Errors**:
   - Check file format support
   - Verify file integrity
   - Use librosa for format conversion

### Getting Help
- Check the logs in `logs/` directory
- Review configuration files
- Run with debug logging enabled

## Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Submit** a pull request

### Development Setup
```bash
git clone <your-fork>
cd environmental-sound-classification
pip install -r requirements.txt
pre-commit install
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{environmental_sound_classification,
  title={Environmental Sound Classification with Deep Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Environmental-Sound-Classification}
}
```

## Acknowledgments

- **ESC-50 Dataset**: Karol J. Piczak for the environmental sound dataset
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing library
- **Streamlit**: Web application framework
- **Hydra**: Configuration management

## Future Work

- [ ] Transformer-based models
- [ ] Multi-modal audio-visual classification
- [ ] Real-time streaming classification
- [ ] Mobile app deployment
- [ ] Additional datasets (UrbanSound8K, AudioSet)
- [ ] Model compression and quantization
- [ ] Federated learning support

---

**Remember**: This is a research demonstration. Use responsibly and ethically!
# Environmental-Sound-Classification
