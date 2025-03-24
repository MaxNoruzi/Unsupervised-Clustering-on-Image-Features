# 📊 Unsupervised Clustering on Image Features

This project performs **unsupervised clustering** on high-dimensional image feature vectors using popular clustering algorithms:  
- **KMeans**
- **DBSCAN**
- **Mean Shift**

It uses **t-SNE** for visualization and **majority-vote label assignment** to evaluate clustering performance with accuracy scores.

---

## 📦 Features

- 📁 Loads precomputed image features (`.npz`) with partial ground-truth domain labels
- 🔍 Implements and evaluates:
  - KMeans (with auto-optimal `k`)
  - DBSCAN (grid search over `eps`, `min_samples`)
  - Mean Shift (bandwidth-based clustering)
- 📈 Calculates **accuracy** using majority-vote conversion of cluster labels
- 📉 Visualizes high-dimensional clusters with **t-SNE**
- 🧪 Supports test and train datasets

---

## 🧰 Tech Stack

- Python 3.x
- NumPy
- scikit-learn
- Matplotlib
- t-SNE (via scikit-learn)

---

