# MNIST Digit Classification (Deep Learning)



Train a feedforward neural network to classify handwritten digits (0–9) from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The notebook includes training/validation curves, per-batch cost decrease visualization, and full evaluation metrics.

## Contents

- **Data:** MNIST load, normalize (0–1), and one-hot labels.
- **Model:** Sequential MLP — Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, softmax).
- **Training:** Adam optimizer, categorical cross-entropy loss, 10 epochs, 10% validation split.
- **Curves:**
  - Loss and accuracy vs epoch (train and validation).
  - Cost decrease with backpropagation: batch-level loss for the first two epochs.
- **Metrics:** Test accuracy, per-class and macro/weighted precision, recall, F1-score, and confusion matrix.

## How to run

1. Install dependencies (TensorFlow, NumPy, Matplotlib, Seaborn, scikit-learn, pandas).
2. Open `mnist.ipynb` in Jupyter or VS Code/Cursor.
3. Restart the kernel and run all cells from the top.

### Dependencies

- Python 3.8+
- `tensorflow` (or `tensorflow-cpu`)
- `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pandas`

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

## Output

- Training/validation loss and accuracy plots.
- Batch-level loss plot (cost decrease within epochs).
- Per-class metrics table and classification report.
- Confusion matrix heatmap and a short summary table of key metrics.

