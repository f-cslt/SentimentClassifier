# SentimentClassifier

A sentiment analysis model for movie reviews using an LSTM (Long Short-Term Memory) neural network, trained on the IMDb dataset.

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python main.py
```

## Architecture

| Layer            | Details                                                    |
| ---------------- | ---------------------------------------------------------- |
| **Embedding**    | Converts word indices to dense 256-D vectors               |
| **LSTM**         | 2-layer LSTM with 256 hidden units, processes the sequence |
| **Mean Pooling** | Masked average over all non-padding timesteps              |
| **Dropout**      | 50% dropout to prevent overfitting                         |
| **Linear**       | Maps pooled representation to a single logit               |

## Dataset

The IMDb dataset contains 50,000 movie reviews:

- **25,000** for training
- **25,000** for testing

Each review is labeled as positive (`1`) or negative (`0`).

## Project Structure

```
OpinionClassifier/
├── main.py            # Entry point
├── dataset.py         # Data loading & preprocessing
├── model.py           # LSTM model architecture
├── train.py           # Training loop & evaluation
└── requirements.txt   # Python dependencies
```

## Dependencies

- PyTorch
- Hugging Face Datasets
