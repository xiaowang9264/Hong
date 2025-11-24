# Optimizers with Fashion-MNIST

This project contains the results of comparative experiments on the **Fashion-MNIST dataset**, using different **CNN architectures** and **optimization algorithms** to evaluate their performance in image classification tasks.

The goal is to explore how various optimizers (e.g., SGD, Adam, RMSprop) affect training dynamics and final accuracy when applied to different CNN models. This study provides insights into model convergence, training stability, and generalization ability under varying optimization strategies.

---

## 📁 Project Structure

```text
Optimizers with Fashion-MNIST/
│
├── Experiment_code/              # Main code for training and evaluation
│   ├── .idea/                    # IDE configuration (IntelliJ/PyCharm)
│   ├── data/                     # Preprocessed or downloaded datasets
│   └── main.py                   # Core script: defines models, optimizers, training loop
│
├── Result_show/                  # Output visualization and result analysis
│   ├── plots/                    # Training curves (loss, accuracy)
│   └── comparison_table.xlsx     # Performance comparison table
│
├── Compare_tabel.docx            # Word document with detailed results
└── README.md                     # This file(Including the experimental environment setup and dependent libraries)

```

---

## 🧪 Experimental Setup

### Dataset
- **Fashion-MNIST**: A 60,000-sample grayscale image dataset consisting of 10 classes (e.g., T-shirt, Trouser, Pullover, etc.), each with 28×28 pixels.
- Used as a drop-in replacement for MNIST with more complex visual patterns.

### Models Tested
We implemented and compared the following CNN architectures:
- **Simple CNN**: 3 convolutional layers + max pooling + fully connected layers.
- **Deeper CNN**: Additional conv layers and batch normalization.
- *(Optional: ResNet-like blocks or other variants)*

Each model was trained with each optimizer, resulting in a total of `n_models × n_optimizers` configurations.

---

## 🚀 How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/xiaowang9264/Hong.git
   cd Optimizers-with-Fashion-MNIST
   ```

2.Navigate to the code directory:
   ```bash
   cd Experiment_code
   ```

3.Install required dependencies (recommended via pip):
   ```bash
   pip install torch torchvision matplotlib numpy pandas
   ```

4.Run the experiment:
   ```bash
   python main.py
   ```
---

## 📊 Results & Analysis
All experimental results are summarized in the Result_show/ folder:
  Plots: Visualize training progress (e.g., loss vs epoch, accuracy trends).
  Comparison Table: Shows test accuracy, training time, and convergence speed across combinations.
Additionally, the Compare_tabel.docx document provides a detailed tabular comparison.

---

## 📝 Future Work
Add learning rate scheduling.
Include regularization techniques (dropout, weight decay).
Test on larger datasets like CIFAR-10.
Implement automated hyperparameter tuning (Bayesian optimization).

---

## 🙌 Acknowledgments
Special thanks to:
Zalando Research for providing the Fashion-MNIST dataset.
PyTorch team for the excellent deep learning framework.

---

## 📄 License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it for educational or research purposes.
