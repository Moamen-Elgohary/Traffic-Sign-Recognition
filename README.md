# Traffic Sign Recognition

> *A misread road sign at speed can be fatal — making reliable traffic sign recognition a core challenge in autonomous driving.*

## Problem & Approach

Real-world signs appear faded, angled, occluded, or poorly lit, and the GTSRB dataset reflects that with 43 classes and significant class imbalance. This project compares four deep learning models — each adding one deliberate change over the last — to find what actually moves the needle.

---

## Dataset

- **Source:** [GTSRB — German Traffic Sign Recognition Benchmark via KaggleHub](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes:** 43 traffic sign categories
- **Files Used:**
  - `Train/` folder + `Train.csv`
  - `Test/` folder + `Test.csv`
  - `Meta/` folder + `Meta.csv`

Images were resized to **32×32** (CNN models) or **96×96** (MobileNet) and normalized to [0, 1].

---

## Project Structure

```
Traffic_Sign_Recognition.ipynb
│
├── Libraries
├── Dataset
├── Setting Paths to Dataset
├── Load & Preprocess Train Data
├── Load & Preprocess Test Data
├── CNN (No Augmentation)
│   ├── Train
│   └── Evaluate
├── CNN with Augmentation
│   ├── Train
│   └── Evaluate
├── Deeper CNN with Augmentation
│   ├── Train
│   └── Evaluate
├── MobileNet (Transfer Learning)
└── Comparison Summary
```

---

## Models & Results

### 1. CNN — No Augmentation

**What was done:** A shallow convolutional neural network was built from scratch with a few Conv2D + MaxPooling layers, trained directly on the raw preprocessed images with no artificial variation introduced. This establishes a clean baseline — it shows what a straightforward CNN can learn from the data as-is, without any tricks.

- **Test Accuracy:** ~83%
- Learns general patterns (shape, color) well.
- Struggles with rare or visually similar classes.
- Serves as the baseline.

---

### 2. CNN — With Augmentation

**What was done:** The same shallow CNN architecture was kept, but the training pipeline was extended with data augmentation — randomly applying rotations, zooms, shifts, and flips to each batch during training. The model never sees the exact same image twice, forcing it to generalize rather than memorize.

- **Training Accuracy:** ~95.77% | **Test Accuracy:** ~83%
- Augmentation improved class-level performance for underrepresented signs.
- Overall accuracy remained similar to baseline but with better robustness.

---

### 3. Deeper CNN — With Augmentation

**What was done:** The architecture was scaled up significantly — more convolutional layers, larger filters, and batch normalization were added to give the model more expressive power. Augmentation was kept from the previous step. This tests whether depth alone, combined with regularization from augmentation, is enough to break through the accuracy ceiling hit by the shallower models.

- **Test Accuracy:** ~97%
- Significant improvement from increased depth and larger filters.
- Strong, balanced performance across nearly all 43 classes.

---

### 4. MobileNet (Transfer Learning)

**What was done:** Instead of training from scratch, MobileNet — a lightweight CNN pretrained on ImageNet — was loaded with its convolutional base frozen, and a custom classification head was added on top. Images were upscaled to 96×96 to match MobileNet's expected input. This tests whether knowledge from a large general-purpose dataset can transfer effectively to the specific domain of traffic signs.

- **Test Accuracy:** ~77%
- High precision/recall on common classes (4, 5, 6, 7, 8, 9, 26).
- Poor performance on several classes (0, 11, 31, 33) — precision/recall below 0.5.
- Macro average: 0.70, indicating class-level imbalance in predictions.

---

## Comparison Summary

| Model | Test Accuracy | Notes |
|---|---|---|
| CNN (no augmentation) | ~83% | Fast baseline; struggles with rare classes |
| CNN + Augmentation | ~83% | Better class robustness |
| **Deeper CNN + Augmentation** | **~97%** | **Best overall — strong across all classes** |
| MobileNet | ~77% | Underperforms; class imbalance issues |

The **Deeper CNN with Augmentation** achieved the best test accuracy of ~97%, outperforming both shallower architectures and MobileNet on this dataset.

---

## Conclusion

The results tell a clear story. Augmentation alone — applied to a shallow network — didn't move the accuracy needle, but it did improve consistency across underrepresented classes, which matters in safety-critical applications. The real leap came from combining augmentation with a deeper architecture: the Deeper CNN hit ~97% accuracy, a strong result for a model trained entirely from scratch on 32×32 images.

The more surprising finding is MobileNet's underperformance at ~77%. Transfer learning is often treated as a silver bullet, but this project shows it has limits. MobileNet's ImageNet-pretrained features, designed for rich natural images, don't map cleanly onto the compact, symbolic structure of traffic signs. Fine-tuning with a frozen base leaves the model unable to fully adapt, and the class imbalance in the dataset compounds the problem.

The key takeaway: **for domain-specific, visually constrained tasks like traffic sign recognition, a well-designed custom CNN with augmentation can outperform heavyweight transfer learning models.** Bigger isn't always better — fit to the problem is.

---

## Libraries Used

```python
numpy, pandas, matplotlib, seaborn
opencv-python (cv2)
tensorflow / keras (Sequential, Conv2D, MaxPooling2D, MobileNet)
scikit-learn (confusion_matrix, classification_report)
kagglehub
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install kaggle kagglehub tensorflow opencv-python scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Traffic_Sign_Recognition.ipynb
   ```
   Run all cells sequentially. The dataset will be downloaded automatically via KaggleHub.

---

## License

### Code
This project is licensed under the [MIT License](LICENSE).

### Dataset
The GTSRB dataset is provided for **non-commercial, research and educational use only** under the terms of the original benchmark organizers. If you use this dataset, please cite:

> J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. *The German Traffic Sign Recognition Benchmark: A multi-class classification competition.* In Proceedings of the IEEE International Joint Conference on Neural Networks, 2011.

The dataset is accessed via [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and is subject to [Kaggle's Terms of Service](https://www.kaggle.com/terms). Use of this dataset for commercial purposes is not permitted without explicit permission from the original dataset authors.
