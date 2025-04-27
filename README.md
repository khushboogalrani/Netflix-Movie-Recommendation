# Netflix Movie Recommendation System

A multi-approach movie recommendation system inspired by Netflix's recommendation principles.  
This project builds and compares multiple models — from classic similarity matching to modern embedding-based techniques — to create a well-rounded understanding of how personalized recommendations are designed.

---

## 🚀 Overview

- Developed and compared **Content-Based Filtering** and **Collaborative Filtering** methods.
- Implemented a range of models from **TF-IDF**, **K-Means Clustering**, **Word2Vec**, and **Node2Vec**, to **Sentence Transformers** and **Matrix Factorization (SVD)**.
- Explored different ways to model movie similarity: text-based, graph-based, and user-behavior-based.
- Designed with **modular, extensible notebooks** — easy to build upon or integrate into hybrid systems.
- Focused on **practical machine learning applications** relevant to recommendation engines.

---

## 🛠 Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- gensim
- nltk
- networkx
- matplotlib
- sentence-transformers
- scikit-surprise
- scipy

---

## 📂 Project Structure

```plaintext
Netflix-Movie-Recommendation/
│
├── collaborative_filtering/
│   ├── Pearson_Correlation_Recommender.ipynb
│   └── SVD_Recommender.ipynb
│
├── content_based_filtering/
│   ├── TFIDF_CosineSimilarity_Recommender.ipynb
│   ├── MiniBatchKMeans_Recommender.ipynb
│   ├── Word2Vec_Recommender.ipynb
│   ├── Node2Vec_Recommender.ipynb
│   ├── SentenceTransformer_Recommender.ipynb
│
├── data/   # Empty placeholder folder
│
├── prepare_data.ipynb
├── README.md
├── requirements.txt
├── .gitignore
```

---

## 📦 Datasets

Datasets are publicly available and can be downloaded from:

- **Netflix Prize Data**: [kaggle.com/netflix-inc/netflix-prize-data](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- **Netflix Movies and TV Shows**: [kaggle.com/shivamb/netflix-shows](https://www.kaggle.com/shivamb/netflix-shows)

> Place downloaded files inside the `/data/` directory.

---

## 🧐 Models Implemented

### Content-Based Filtering

| Model | Approach |
|:------|:---------|
| **TF-IDF + Cosine Similarity** | Matches movies based on plot descriptions. |
| **MiniBatch K-Means Clustering** | Clusters movies into thematic groups for recommendations. |
| **Word2Vec Embeddings** | Learns semantic similarity from metadata like titles, genres, and cast. |
| **Node2Vec Graph Embeddings** | Models movies as a graph of relationships and learns node embeddings. |
| **Sentence Transformers** | Generates deep semantic representations of movie plots. |

### Collaborative Filtering

| Model | Approach |
|:------|:---------|
| **Pearson Correlation** | Recommends based on similarity in user ratings. |
| **SVD (Singular Value Decomposition)** | Matrix factorization to uncover latent user and movie features. |

---

## 🌟 Highlights

- 🔍 **Multi-model exploration**: Text similarity, clustering, graph embeddings, and collaborative filtering.
- 🛠️ **Applied Machine Learning**: End-to-end feature engineering, model training, and evaluation.
- ✍️ **Clear modular design**: Each method implemented in its own notebook for clarity and ease of extension.

---

## ✅ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Netflix-Movie-Recommendation.git
   cd Netflix-Movie-Recommendation
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download datasets and place them inside `/data/`.

5. Open and run the notebooks individually.

---

## 📄 License

MIT License.

---

> ⭐ If you find this project interesting, feel free to star it or fork it.  
> 🚀 Always open to collaborations or conversations about machine learning, NLP, and recommender systems!

