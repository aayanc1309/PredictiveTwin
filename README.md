# AI-Driven PredictiveTwin

AI-powered **Digital Twin system** for smart manufacturing that simulates machine behavior, predicts failures, and generates **LLM-based maintenance alerts**.

---

## 🚀 Features

* Predictive Maintenance (RF, XGBoost, MLP)
* Digital Twin Simulation (SimPy-based factory)
* Real-time Inference Pipeline
* LLM-powered Alert Generation (Gemini)
* Streamlit Dashboard + MLflow Tracking

---

## 📊 Model Performance

| Model                  | Accuracy | Precision | Recall     | F1 Score   | ROC-AUC    |
| ---------------------- | -------- | --------- | ---------- | ---------- | ---------- |
| XGBoost (Optuna)       | **0.97** | 0.7668    | 0.8639     | **0.8070** | 0.9616     |
| Random Forest (Optuna) | 0.9585   | 0.7150    | **0.8721** | 0.7695     | 0.9677     |
| MLP (Optuna)           | 0.9055   | 0.6222    | 0.8943     | 0.6686     | **0.9691** |

✅ Best model selected based on **F1-score (XGBoost)**

---

## 🏗️ Project Structure

```
smart_manufacturing/
├── data/              # dataset + preprocessing
├── models/            # training, tuning, evaluation
├── simulation/        # digital twin (SimPy)
├── llm/               # alert generation
├── dashboard/         # Streamlit UI
├── logs/              # alert logs
├── mlruns/            # MLflow tracking
├── main.py            # pipeline entry point
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Set Gemini API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

Place dataset:

```
data/ai4i2020.csv
```

---

## ▶️ Run

Full pipeline:

```bash
python main.py
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

MLflow UI:

```bash
mlflow ui
```

---

## 🔄 System Flow

Simulation → Sensor Data → ML Model → Failure Prediction → LLM Alert → Dashboard

---

## 🧠 Tech Stack

Python • Scikit-learn • XGBoost • PyTorch • SimPy • Streamlit • Plotly • MLflow • Gemini API

---

## 📌 Summary

End-to-end **Digital Twin + AI system** that transforms predictive maintenance into a **real-time, explainable, and interactive solution** for smart manufacturing.
