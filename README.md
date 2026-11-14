# Heart Disease Prediction API

A FastAPI app that serves predictions from a Random Forest classifier trained on the Heart Disease dataset. The app is Dockerized and deployed on Render. The focus of this project is Docker and deployment, not model accuracy.

**Live API (Swagger UI):** https://heart-disease-api-4el0.onrender.com/docs

The free Render instance sleeps after 15 minutes without requests, so the first request can take about a minute.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app and endpoints
│   └── schemas.py       # Pydantic input and output models
├── data/
│   └── heart.csv        # Heart Disease dataset (Kaggle, johnsmith88)
├── model/
│   └── heart_model.joblib
├── train_model.py       # Trains the model and saves it with joblib
├── sample.json          # Example request body
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Dataset and Model

- Dataset: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (1025 rows, 13 features, binary target).
- Model: `RandomForestClassifier` (scikit-learn), saved to `model/heart_model.joblib` with joblib.
- The dataset contains many duplicate rows, so the test accuracy is optimistic. This project is about the API, Docker and deployment.

## Input Features

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

Each field is validated by a Pydantic model (`app/schemas.py`). A value outside the allowed range returns a `422` error.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check, returns `{"status": "ok"}` |
| GET | `/info` | Model type and feature list |
| POST | `/predict` | Returns `{"heart_disease": true}` or `{"heart_disease": false}` |

## Run Locally with Docker

```bash
docker-compose build
docker-compose up
```

Then open http://localhost:8000/docs for the Swagger UI.

## Example Requests

```bash
curl https://heart-disease-api-4el0.onrender.com/health
curl https://heart-disease-api-4el0.onrender.com/info
curl -X POST https://heart-disease-api-4el0.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

Example response:

```json
{"heart_disease": false}
```

## Retrain the Model

```bash
pip install -r requirements.txt
python train_model.py
```

## Deployment on Render

- Service type: Web Service, language: Docker, instance type: Free
- Branch: `main`, root directory: repository root (the `Dockerfile` is at the root)
- Health check path: `/health`
- The `Dockerfile` starts uvicorn on the `PORT` that Render provides (default `8000`)