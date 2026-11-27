# ML Inference Service

Production-style FastAPI service serving a scikit-learn Breast Cancer
classification model (trained in Day 63). Containerized with Docker and
deployed via GitHub Actions → Artifact Registry → Cloud Run (Day 64).

## Architecture

```
Developer workstation
        │  git push / pull request
        ▼
GitHub repository ──► GitHub Actions (CI/CD)
        │                lint → test → build → scan → publish → deploy
        ▼
Google Artifact Registry (versioned Docker images)
        │
        ▼
Google Cloud Run (HTTPS, auto-scaling, revisions)
        │
        ▼
Live ML API — /health, /metadata, /predict
```

## Project structure

```
ml-inference-service/
├── app/
│   ├── __init__.py
│   └── main.py            # FastAPI app (lifespan loads model once)
├── artifacts/
│   └── breast_cancer_model.joblib
├── scripts/
│   └── train_model.py     # retrain pipeline -> artifacts/
├── tests/
│   └── test_api.py        # pytest + FastAPI TestClient
├── requirements.txt       # pinned production deps
├── requirements-dev.txt   # + pytest, httpx, ruff
├── Dockerfile             # multi-stage, non-root, healthcheck
├── .dockerignore
└── compose.yaml           # local multi-container dev
```

## Local development

```bash
# train / retrain the model
python scripts/train_model.py

# install deps and run tests
pip install -r requirements-dev.txt
pytest -v
ruff check app scripts tests

# run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs

## Docker

```bash
docker build -t ml-inference-api:v1 .
docker run -d --name ml-api -p 8000:8080 ml-inference-api:v1

curl http://localhost:8000/health
curl http://localhost:8000/docs

docker logs -f ml-api
docker stats
```

Or with Compose:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f
docker compose down
```

## CI/CD

Workflows live at the repo root in `.github/workflows/`:

| Workflow      | Trigger                          | What it does                              |
| ------------- | -------------------------------- | ----------------------------------------- |
| `ci.yml`      | push/PR to `main`                | lint → pytest → docker build              |
| `deploy.yml`  | push/PR to `main`                | test → OIDC auth → build+push → Cloud Run |
| `security.yml`| push/PR, weekly schedule         | Trivy vulnerability scan of the image     |

The deploy job only runs on pushes to `main` (PRs test but never deploy).
It deploys by **image digest**, not a mutable `latest` tag.

### GitHub repository variables (Settings → Secrets and variables → Actions → Variables)

| Variable           | Example                                        |
| ------------------ | ---------------------------------------------- |
| `GCP_PROJECT_ID`   | `my-ml-project`                                |
| `GCP_REGION`       | `asia-south1`                                  |
| `GCP_REPOSITORY`   | `ml-containers`                                |
| `GCP_SERVICE`      | `ml-inference-api`                             |
| `GCP_WIF_PROVIDER` | `projects/<num>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `GCP_DEPLOYER_SA`  | `github-deployer@<project>.iam.gserviceaccount.com` |

### One-time Google Cloud setup

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com

gcloud artifacts repositories create ml-containers \
  --repository-format=docker --location=asia-south1

gcloud iam workload-identity-pools create github-pool --location=global

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --issuer-uri="https://token.actions.githubusercontent.com/" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository == 'OWNER/REPO' && assertion.ref == 'refs/heads/main'"

gcloud iam service-accounts create github-deployer

# allow the GitHub repo identity to impersonate the deployer SA
# (see README notes / Google docs for the add-iam-policy-binding command)
```

Grant the deployer SA only: Artifact Registry Writer (on the repo),
Cloud Run Developer, and Service Account User on the runtime identity.
Never use project Owner.

## Cloud Run contract

- App listens on `0.0.0.0` (required — traffic comes from outside the container).
- Port comes from the `PORT` env var (Cloud Run injects it, default 8080).
- Dockerfile `CMD` already handles this:
  `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}`

## Deployment safety notes

- Deploys are immutable revisions — Cloud Run keeps old revisions for rollback.
- Traffic splitting / canary:
  `gcloud run services update-traffic ml-inference-api --region asia-south1 --to-revisions OLD=95,NEW=5`
- Rollback: point 100% traffic back to the previous revision.
- Set `--max` instances and a billing budget alert; clean up test resources.
- Never commit secrets. Runtime secrets belong in Secret Manager, not the image.
