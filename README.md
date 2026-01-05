# Emotion Detection Using DistilBERT with Robust Text Normalization

## Project Overview
This project implements an **emotion classification system** using a **pretrained DistilBERT transformer**, fine-tuned on labeled text data.  
The system is designed to be **robust to noisy input**, including slang and emojis, by applying a custom text normalization pipeline prior to model training and inference.

The project evaluates model performance on both **clean** and **noisy** datasets and demonstrates how additional fine-tuning on noisy data improves generalization and stability.

---

## Objectives
- Build an emotion classification model using a pretrained Transformer
- Normalize real-world text containing slang and emojis
- Evaluate robustness against noisy input
- Improve performance through targeted fine-tuning
- Map predictions to human-readable emotion labels

---

## Model Architecture
- **Base model:** DistilBERT (pretrained, uncased)
- **Task:** Multi-class emotion classification
- **Training approach:** Transfer learning with fine-tuning
- **Framework:** Hugging Face Transformers

DistilBERT is a lightweight transformer that retains most of BERT’s performance while being faster and more efficient, making it suitable for real-world NLP applications. :contentReference[oaicite:0]{index=0}

---

## Data Pipeline

### 1. Text Normalization
A custom `TextNormalizer` class standardizes raw input text by:
- Converting text to lowercase
- Replacing slang words using a JSON dictionary
- Converting emojis into semantic text labels
- Removing extra whitespace

This step significantly improves model robustness on informal and noisy text.

### 2. Dataset Preparation
- Input files: `train.txt`, `val.txt`, `test.txt`, `noisy_test.txt`
- Labels are mapped to numeric IDs
- Clean and noisy datasets are evaluated separately
- Custom `labels.json` is generated for consistent label decoding

---

## Training & Evaluation

### Training Strategy
1. Fine-tune DistilBERT on clean training data
2. Evaluate on clean and noisy test sets
3. Perform short additional fine-tuning on noisy data
4. Re-evaluate to measure robustness improvement

### Metrics Used
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

Macro averaging ensures equal importance for all emotion classes.

---

## Key Results
- Strong performance on clean test data (macro F1 ≈ 0.88, accuracy ≈ 93%)
- Significant performance drop on noisy data before adaptation
- Substantial recovery after fine-tuning on noisy data:
  - Noisy F1-score improved from ~77% to ~86%
  - Clean test performance also improved slightly
- Demonstrates effective robustness learning

---

## Emotion Prediction Examples
- "I am sad." → **sadness** (0.97)
- "This is the best day ever!" → **joy** (0.70)
- "I feel anxious and worried." → **fear** (0.95)
- "This makes me angry!" → **anger** (0.99)

Predictions are mapped using a **custom label JSON**, not hardcoded class names.

---

## Technologies Used
- Python
- Hugging Face Transformers
- DistilBERT
- scikit-learn
- pandas, numpy
- JSON-based text normalization

---

## Skills Demonstrated
- Transformer-based NLP modeling
- Transfer learning and fine-tuning
- Text normalization for noisy real-world data
- Robust evaluation across clean and noisy datasets
- Custom label mapping and inference pipelines
- Reproducible experiment design

---

## Limitations
- Slang and emoji dictionaries require manual maintenance
- Performance may degrade on unseen emotion categories
- No multilingual support in current version

---

## Future Enhancements
- Add multilingual emotion classification
- Introduce data augmentation for rare emotions
- Compare against RoBERTa and DeBERTa
- Deploy as a REST API or web demo
- Integrate confidence calibration for production use

---

## Conclusion
This project demonstrates a **production-aware NLP pipeline** that combines transformer models with practical text normalization to handle noisy, real-world input.  
It highlights strong understanding of modern NLP workflows, robustness evaluation, and applied deep learning.
