# AstroVision – Exoplanet AI Detection

**AstroVision** is a collaborative project that combines **machine learning** and **interactive visualization** to identify and classify potential exoplanets from NASA missions (Kepler, TESS, and K2).
It integrates a **Python backend** powered by a LightGBM classifier and a **React-based frontend** that allows users to intuitively explore classification results.

---

## 🌌 Project Overview

Our goal is to democratize exoplanet discovery by merging advanced data processing with an accessible, game-like interface.
The backend analyzes celestial data to classify each object as a **Planet**, **Candidate Planet**, or **Not Planet**, while the frontend presents these results visually and interactively.

---

## ⚙️ Backend – Machine Learning Model (Python)

The ML model extracts attributes from NASA datasets (Kepler, TESS, K2) using features derived from:

* **Transit Method**
* **Radial Velocity**
* **Astrometry**

Data were split into **80% training** and **20% testing**, and a **LightGBM decision tree model** was used for classification.
A **linear regression step** converts the model’s output probabilities into three categories:

| Probability Range | Classification   |
| ----------------- | ---------------- |
| P < 50%           | Not Planet       |
| 50% ≤ P < 95%     | Candidate Planet |
| P ≥ 95%           | Planet           |

### **Performance Summary**

| Mission | Accuracy | Precision | Recall | F1-score | Predicted Positives |
| ------- | -------- | --------- | ------ | -------- | ------------------- |
| Kepler  | 92.6%    | 79.7%     | 99.0%  | 88.5%    | 4,388               |
| K2      | 82.1%    | 60.0%     | 84.0%  | 69.9%    | 6,060               |
| TESS    | 79.4%    | 36.8%     | 62.6%  | 46.3%    | 2,044               |

These metrics show that AstroVision’s model is **highly deterministic**, producing fewer ambiguous candidates and a stronger separation between planets and non-planets.

---

## 💻 Frontend – React Interface

The **frontend**, developed in **React (JavaScript, JSX, CSS)**, visualizes the ML results and provides an educational interface that engages users of different expertise levels.

* **Beginner mode:** simplified, educational visualization of planet candidates.
* **Advanced mode:** detailed analytics, probability thresholds, and confusion matrices for research use.

### 🎨 Interactive Prototype

To preview the **AstroVision** front-end interface and animations, you can explore our live **Figma prototype** here:

👉 [View the AstroVision UI/UX Animation on Figma](https://www.figma.com/proto/GOgsWQN1HcpZFNVrXogtJc/Astrovision?page-id=0%3A1&node-id=5-2&viewport=418%2C-211%2C0.18&t=zvKqkxmrqhbvQiXD-1&scaling=min-zoom&content-scaling=fixed&starting-point-node-id=5%3A2&show-proto-sidebar=1)

This prototype demonstrates the **animated navigation**, **planet visualization transitions**, and **user interaction flow** designed for both beginner and advanced modes.

### **Frontend Status**

* The React interface is fully functional with routing, components, and styling.
* The backend API (Python) handles data ingestion and classification independently.
* Integration between React and Python API endpoints is planned for the next stage.

---

## 🧠 AI Tools and Libraries

**Backend:**

* `scikit-learn`, `lightgbm`, `numpy`, `pandas`, `imbalanced-learn`
* `matplotlib`, `seaborn`, `json`, `flask` (for serving APIs)
* **Python 3.10+**

**Frontend:**

* `React`, `Vite`, `Node.js`, `Axios`, `React Router`, `Chart.js`

**AI Assistance:**

* ChatGPT-5 and OpenAI Codex were used for code generation, documentation, and data processing assistance.

---

## 🧹 Repository Structure

```
├── src/                # ML scripts and training modules
├── artifacts/          # Generated confusion matrices and reports
├── data/               # Processed datasets
├── frontend/           # React user interface
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment configuration
└── README.md           # This file
```

---

##  Running the Project

### Backend (Python)

```bash
# Create environment
conda env create -f environment.yml
conda activate exoplanets

# Run training pipeline
python -m src.exo_tabular --mode train --split cross-mission --oversample
```

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

The React app will run on `http://localhost:5173` (by default) and will connect to the Flask API once integration is finalized.

---

## 🚀 Next Steps

* Connect React frontend with Python backend via REST API.
* Deploy unified app (frontend + backend) on **Vercel** and **Render/Heroku**.
* Enhance visualization of probability distributions and planetary discovery statistics.

---

## 👩‍🚀 Authors

* **Marcus Vinicius Leal de Carvalho** – Machine Learning, NASA Dataset Integration
* **Madu** – Frontend Development (React UI/UX)
* **AstroVision Team** – Data Science, Research & Design

---

## 🌠 License

This project was developed as part of the **NASA Space Apps Challenge 2025**.
All datasets are from the **NASA Exoplanet Archive** and are publicly available.
