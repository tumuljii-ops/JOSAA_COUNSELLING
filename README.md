# 🎓 JoSAA Counselling Rank Predictor & College Recommendation System

An end-to-end Machine Learning powered web application that predicts JoSAA counselling outcomes and recommends colleges/branches based on JEE rank, category, gender, and quota.

Built using **XGBoost**, **FastAPI**, **Streamlit**, **Docker**, and deployed on the cloud using **Render** and **Streamlit Cloud**.

---

# 🚀 Live Demo

## 🌐 Frontend Application
https://josaacounselling-ehfdd2knpcatt5bi4nkvbc.streamlit.app/

## ⚡ FastAPI Backend Docs
https://josaa-counselling.onrender.com/docs

---

# 📌 Project Overview

This project predicts college admission possibilities using historical JoSAA counselling data containing more than **50,000+ admission records**.

The system:
- takes user rank and reservation details
- predicts expected closing ranks
- classifies colleges into:
  - High Chance
  - Medium Chance
  - Dream Colleges
- returns top recommendations in real time

The complete ML pipeline is deployed as a production-style system with:
- REST APIs
- frontend UI
- Docker containers
- cloud deployment

---

# 🧠 Machine Learning Pipeline

## Dataset
- Multi-year JoSAA counselling dataset
- 50,000+ records
- Features include:
  - Institute
  - Program
  - Quota
  - Gender
  - Seat Type
  - Opening Rank

---

## ML Models Used

| Model | Performance |
|---|---|
| Linear Regression | R² = 0.84 |
| XGBoost Regressor | R² = 0.91+ |

XGBoost significantly improved prediction quality through:
- feature engineering
- hyperparameter tuning
- k-fold cross validation

---

# ⚙️ System Architecture

```text
Frontend (Streamlit)
        ↓
POST Request
        ↓
FastAPI Backend
        ↓
Pydantic Validation
        ↓
XGBoost Model Inference
        ↓
Recommendation Engine
        ↓
JSON Response
        ↓
Frontend Table Output
```

---

# 🏗️ Tech Stack

## Machine Learning
- XGBoost
- Scikit-learn
- Pandas
- NumPy

## Backend
- FastAPI
- REST APIs
- Pydantic

## Frontend
- Streamlit

## Deployment & DevOps
- Docker
- Docker Compose
- Render
- Streamlit Cloud

---

# 🐳 Dockerized Architecture

The project is fully containerized using Docker.

## Containers

### Backend Container
- FastAPI server
- XGBoost inference
- REST APIs

### Frontend Container
- Streamlit UI
- User interaction layer

Docker Compose was used to:
- manage multi-container setup
- create shared networking
- automate service startup

---

# 📊 Features

✅ Real-time college prediction  
✅ ML-powered recommendation engine  
✅ FastAPI backend APIs  
✅ Interactive Streamlit frontend  
✅ Dockerized deployment  
✅ Cloud-hosted inference pipeline  
✅ Publicly accessible web application  

---

# 📂 Project Structure

```text
JOSAA_COUNSELLING/
│
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── model.pkl
│   ├── columns.pkl
│   └── reference_df.pkl
│
├── Frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml
└── README.md
```

---

# 🔥 Running Locally

## Clone Repository

```bash
git clone <https://github.com/tumuljii-ops/JOSAA_COUNSELLING.git>
```

---

## Run Using Docker Compose

```bash
docker compose up --build
```

---

## Open Applications

### Frontend
```text
http://localhost:8501
```

### Backend API Docs
```text
http://localhost:8000/docs
```

---

# ☁️ Deployment

## Backend Deployment
- Platform: Render
- Deployment Type: Dockerized FastAPI Service

## Frontend Deployment
- Platform: Streamlit Community Cloud

---

# 📈 Future Improvements

- Add IIT counselling support
- Improve recommendation quality using institute rankings
- Add branch preference weighting
- Add historical trend analysis
- Add authentication and user profiles
- Deploy using Kubernetes for scalability

---

# 🧑‍💻 Author

**Tumul Singh**

- NIT Kurukshetra
- B.Tech Information Technology

---

# ⭐ Key Learnings

Through this project I learned:

- end-to-end ML system design
- FastAPI backend engineering
- REST API development
- Docker & Docker Compose
- cloud deployment workflows
- ML inference pipelines
- frontend-backend integration
- production-style architecture

---

# 📜 License

This project is for educational and research purposes.
