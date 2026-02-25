# Clinical NLP Extraction API

> Scalable, production-oriented clinical NLP system for transforming unstructured medical text into structured, model-ready signals.

---

## 1. Problem

Clinical insights are frequently embedded in **unstructured transcriptions**, limiting their usability for:

- Large-scale analytics  
- Clinical decision support systems  
- Downstream machine learning pipelines  

Manual extraction is **non-scalable, error-prone, and expensive**.

---

## 2. Solution

This system provides a **low-latency, API-driven pipeline** that:

- Extracts clinically relevant entities (symptoms, negations)  
- Classifies medical specialty using transformer models  
- Converts free-text → structured JSON outputs  
- Deploys as a **containerized, horizontally scalable service**  

---

## 3. Key Impact

- Enables **real-time clinical text structuring**  
- Reduces manual chart review workload  
- Provides **ML-ready features** for downstream healthcare models  
- Demonstrates **production-grade MLOps and system design**  

---

## 4. System Design

### High-Level Flow
8. Scalability & Engineering Considerations
Designed For

Stateless API → horizontal scaling

Container orchestration (Kubernetes-ready)

Batch + real-time inference support (extensible)

Optimization Opportunities

Model quantization (reduce latency)

GPU-backed inference endpoints

Request batching for throughput improvement

9. Reproducibility

Deterministic preprocessing pipeline

Clear data lineage (raw/ → processed/)

Docker ensures consistent runtime environment

Modular training + inference separation

10. Repository Structure
clinical-nlp-extraction-api/
├── data/
│   ├── raw/                 # Source dataset (MTSamples)
│   └── processed/           # Cleaned data
├── models/                  # Model artifacts
├── src/
│   ├── api/                 # FastAPI routes
│   ├── model/               # Training + inference
│   └── utils/               # Preprocessing logic
├── Dockerfile
├── requirements.txt
└── README.md

11. Local Development
git clone https://github.com/AidanColvin/clinical-nlp-extraction-api.git
cd clinical-nlp-extraction-api

# Dataset
MTSamples on Kaggle. This dataset provides roughly 5,000 scraped medical transcription records across various medical specialties without requiring strict HIPAA data use agreements.

# Build + run
docker build -t clinical-nlp-api .
docker run -p 8000:8000 clinical-nlp-api

12. Future Work

Add MLflow-based experiment tracking

Implement CI/CD pipeline (GitHub Actions)

Expand entity schema (medications, procedures, diagnoses)

Integrate FHIR-compatible outputs

Add monitoring + alerting for model drift

13. Why This Project Stands Out

This project demonstrates:

End-to-end ownership: data → model → API → deployment

Strong system design and scalability awareness

Practical application of transformer-based NLP in healthcare

Production-oriented thinking (latency, reproducibility, modularity)
