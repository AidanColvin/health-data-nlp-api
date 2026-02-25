Clinical NLP Extraction API
Overview

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for clinical Natural Language Processing (NLP). It uses a fine-tuned language model to process unstructured medical transcriptions. The model classifies the clinical specialty and extracts key medical entities. The application is served via a FastAPI backend and containerized using Docker for scalable cloud deployment.
Architecture

    Data: MTSamples dataset (Kaggle).

    Model: Fine-tuned ClinicalBERT / LLaMA.

    Backend: FastAPI (Python).

    Containerization: Docker.

    Deployment: AWS / Google Cloud Platform (GCP).

Dataset Setup

    Download the mtsamples.csv dataset from Kaggle.

    Place the file in the /data/raw/ directory.

Quick Start

Run the following commands to build and start the Docker container locally.
Bash

# Clone the repository
git clone https://github.com/AidanColvin/clinical-nlp-extraction-api.git
cd clinical-nlp-extraction-api

# Build the Docker image
docker build -t clinical-nlp-api .

# Run the container
docker run -p 8000:8000 clinical-nlp-api

API Usage

Once the container is running, the API is accessible at http://localhost:8000.

Endpoint: /predict
Method: POST
Payload:
JSON

{
  "transcription": "Patient presents with acute knee pain and swelling. Denies chest pain or shortness of breath."
}

Response:
JSON

{
  "specialty": "Orthopedic",
  "entities": {
    "symptoms": ["knee pain", "swelling"],
    "negated_symptoms": ["chest pain", "shortness of breath"]
  },
  "confidence_score": 0.94
}

Directory Structure
Plaintext

clinical-nlp-extraction-api/
├── data/
│   ├── raw/                 # Unprocessed MTSamples data
│   └── processed/           # Cleaned data for model training
├── models/                  # Saved weights and tokenizers
├── src/
│   ├── api/                 # FastAPI routes and schemas
│   ├── model/               # PyTorch training and inference scripts
│   └── utils/               # Text cleaning and preprocessing functions
├── Dockerfile               # Container build instructions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
