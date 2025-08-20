# 🐦 Twitter Sentiment Analysis

A production-ready pipeline to classify tweets as **Positive** or **Negative** using **Logistic Regression** with **TF-IDF** features.  
Built for clarity, reproducibility, and quick reuse (CLI + script + notebook).

> ✅ Current test accuracy: **~86%** (F1 ≈ 0.86) on a balanced subset  
> 🔧 Stack: Python, scikit-learn, NLTK, pandas, NumPy

---

## ✨ Features

- End-to-end workflow: **load → clean → stem → vectorize (TF-IDF) → train → evaluate → save**
- Reusable artifacts: saves **`models/trained_model.sav`** and **`models/vectorizer.sav`**
- Simple **one-tweet prediction** with consistent preprocessing
- Balanced training via **upsampling** (for positive/negative)
- Clear evaluation: **confusion matrix + classification report**

---

## 🛠️ Tech Stack

<p align="center">
  <!-- Core -->
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NLTK-154F9C?style=for-the-badge&logo=python&logoColor=white" alt="NLTK" />
</p>

<p align="center">
  <!-- Tools -->
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git" />
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  <img src="https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VS Code" />
</p>

---

## ⚙️ Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
