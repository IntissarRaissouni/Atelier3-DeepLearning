#  Deep Learning – Lab 3  
## Sequence Models & Transformers for NLP
---

## 1. Introduction

Natural Language Processing (NLP) is a key field of artificial intelligence that enables machines to understand, analyze, and generate human language.  
With the rise of deep learning, advanced architectures such as **Recurrent Neural Networks (RNNs)** and **Transformers** have significantly improved NLP performance.

This lab explores the application of deep learning models for NLP through two main tasks:

- **Part 1**: Arabic text regression using sequence models (LSTM, GRU)
- **Part 2**: Text generation using a Transformer-based model (GPT-2)

---

## 2. Part 1 – Arabic Text Regression Using Sequence Models

### 2.1 Dataset Collection

Arabic text data related to **cybersecurity** was collected and prepared in CSV format.  
Each text sample was assigned a **relevance score between 0 and 10**, representing its importance.

**Dataset structure:**

| Column | Description |
|------|------------|
| `Text` | Original Arabic text |
| `Score` | Relevance score (0–10) |

The final dataset contains **144 text samples**, which is sufficient for experimentation with sequence models in an academic context.

---

### 2.2 Text Preprocessing

To ensure clean and consistent input data, the following preprocessing steps were applied:

- Removal of non-Arabic characters
- Normalization of whitespace
- Text cleaning using regular expressions

A new column `Clean_Text` was created and used for model training.

---

### 2.3 Tokenization and Padding

Since neural networks require numerical input, the cleaned texts were transformed into sequences of integers using a tokenizer.

- Vocabulary size: **10,000 words**
- Maximum sequence length: **100 tokens**
- Padding applied to ensure uniform sequence lengths

The dataset was then split into:
- **80% training data**
- **20% testing data**

---

### 2.4 PyTorch Dataset and DataLoader

The processed data was converted into PyTorch tensors and wrapped into custom `Dataset` and `DataLoader` classes.  
This allowed efficient batching and shuffling during training.

---

### 2.5 LSTM Model Architecture

An **LSTM (Long Short-Term Memory)** model was implemented using PyTorch.  
The architecture consists of:

- An embedding layer
- An LSTM layer
- A fully connected output layer

**Task type**: Regression  
**Loss function**: Mean Squared Error (MSE)  
**Optimizer**: Adam  

---

### 2.6 LSTM Training and Evaluation

The LSTM model was trained for **10 epochs**.  
The training loss showed a stable decreasing trend, indicating effective learning.

**Evaluation metrics on the test set:**

| Metric | Value |
|------|------|
| MAE | 2.71 |
| RMSE | 3.13 |
| MSE | 9.81 |

These results show that the model is capable of reasonably predicting relevance scores despite the limited dataset size.

---

### 2.7 GRU Model

A **GRU (Gated Recurrent Unit)** model was also implemented for comparison.  
GRU has a simpler structure than LSTM and often converges faster.

**GRU evaluation results:**

| Metric | Value |
|------|------|
| MAE | 2.76 |
| RMSE | 3.18 |
| MSE | 10.13 |

---

### 2.8 Model Comparison

| Model | MAE | RMSE | MSE |
|------|-----|------|-----|
| LSTM | 2.71 | 3.13 | 9.81 |
| GRU | 2.76 | 3.18 | 10.13 |

**Observation:**  
The LSTM model slightly outperformed the GRU model on this dataset and was therefore selected as the best-performing model.

---

## 3. Part 2 – Transformer-Based Text Generation (GPT-2)

### 3.1 Transformer Models

Transformers rely on **self-attention mechanisms** instead of recurrence, allowing them to model long-range dependencies efficiently.  
In this part, a pre-trained **GPT-2** model was used for text generation.

---

### 3.2 Model Loading and Fine-Tuning

The GPT-2 model and tokenizer were loaded using the HuggingFace `transformers` library.  
A small custom dataset of sentences related to technology and cybersecurity was used to fine-tune the model.

The model was trained for **3 epochs**, achieving a final training loss of approximately **4.76**, indicating successful fine-tuning.

---

### 3.3 Text Generation

After fine-tuning, the model was used to generate new text from a given prompt.

**Prompt:**
Cybersecurity is

**Generated output example:**

> *Cybersecurity is the fundamental concept behind the Internet, and the future is not just about government control of the Internet.*

This result demonstrates that the model can generate coherent and meaningful text.

---

## 4. Tools and Technologies

- Python
- PyTorch
- HuggingFace Transformers
- TensorFlow Keras (for tokenization only)
- Kaggle Notebook
- GitHub

---

## 5. Conclusion

This lab demonstrated the application of deep learning techniques to NLP tasks using both sequence models and Transformer architectures.  
LSTM and GRU models proved effective for Arabic text regression, while the Transformer-based GPT-2 model successfully generated coherent text.

Overall, this lab provided practical experience with modern NLP pipelines and deep learning frameworks.

---


