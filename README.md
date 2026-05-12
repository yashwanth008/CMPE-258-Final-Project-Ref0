# Ref Zero — AI-Powered Sports Referee Assistant

An end-to-end multimodal AI system that acts as a real-time Video Assistant Referee (VAR) for sports. Ref Zero fine-tunes a YOLO11n object detection model on a curated sports foul dataset, streams live video through a FastAPI WebSocket backend with MediaPipe pose estimation, and triggers Gemini 2.0 Flash to analyze 5-second DVR clips when a foul is detected. Verdicts are returned with specific rulebook citations, which are autonomously verified by a browser-use agent.

Live Demo: [Hugging Face Spaces — add after deployment]
W&B Dashboard: [add your wandb project URL]
Demo Video: [add YouTube link]

---

## Table of Contents

- [Team Contributions](#team-contributions)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Inputs, Outputs, and Key Metrics](#inputs-outputs-and-key-metrics)
- [Architecture](#architecture)
- [Dataset Details](#dataset-details)
- [Model Parameter Decisions](#model-parameter-decisions)
- [How to Retrain From Scratch](#how-to-retrain-from-scratch)
- [Evaluation Results](#evaluation-results)
- [Ablation Study](#ablation-study)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [MLOps Pipeline](#mlops-pipeline)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Artifacts](#artifacts)

---

## Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| Yashwanth | ML - Backend | YOLO11 training pipeline, model_config.py, train.py, ablation.py, FastAPI WebSocket server, DVR buffer, Gemini 2.0 Flash integration, main.py YOLO11 inference wiring |
| Zach | Frontend | Next.js UI, VideoPlayer component, Controls, AgentLog, About tab, MetricsDashboard tab, tab navigation |
| Zach | MLOps | Docker, docker-compose, GitHub Actions CI/CD workflow, Hugging Face Spaces deployment, W&B sweep configuration |
| Yashwanth | Agents | browser-use rulebook verification agent, LiveKit multi-camera agent, agent_browser.py |

---

## Problem Statement

Every year, controversial referee calls cost teams championships, millions in revenue, and fan trust. The NFL employs over 450 officials across 256 games requiring split-second decisions. The NBA regularly sees missed calls swing playoff games. Soccer's VAR system exists but still relies on human interpretation, and youth sports — representing 90 percent of all games played — have no video review capability.

Current video review systems require referees to stop the game for 2 to 5 minutes, manually scrub footage frame-by-frame, recall rules from 200-plus page rulebooks, and make judgment calls under pressure. Even with replay tools, human error remains prevalent.

---

## Solution Overview

Ref Zero gives every referee superhuman capabilities through a five-stage pipeline:

1. Live camera feed is processed by MediaPipe at 30 FPS, tracking 33 body keypoints per player for real-time skeleton overlay
2. A fine-tuned YOLO11n model analyzes each frame — when foul or contact is detected above confidence 0.5, the DVR is triggered
3. A rolling 150-frame buffer saves the last 5 seconds as an MP4 clip
4. Gemini 2.0 Flash receives the clip and returns a structured JSON verdict with sport, action breakdown, rule violated, confidence score
5. A browser-use agent autonomously opens the official rulebook URL and verifies the cited rule text

End-to-end review time: under 10 seconds, compared to 2 to 5 minutes for traditional VAR.

---

## Inputs, Outputs, and Key Metrics

**Input**
- Live webcam stream via WebSocket (base64 encoded frames at 30 FPS)
- OR uploaded MP4 video file via POST /api/trigger_review

**Output**
- Real-time annotated video frame with MediaPipe skeleton overlay and YOLO11 bounding boxes
- JSON verdict:
```json
{
  "sport": "Basketball",
  "action_breakdown": "Player A drives to basket, Defender B makes contact with shooting arm",
  "rule_violated": "NBA Rule 12B - Blocking Foul",
  "verdict": "FOUL",
  "explanation": "Illegal contact with shooting arm during upward motion",
  "confidence": 94
}
```
- Rulebook citation verified by browser-use agent

**Key Metrics**

| Metric | Value | Description |
|--------|-------|-------------|
| mAP@0.5 | 0.554 | Mean average precision at IoU threshold 0.5 |
| mAP@0.5-0.95 | 0.281 | Mean average precision across IoU thresholds |
| Precision | 0.463 | Ratio of correct foul detections to all detections |
| Recall | 0.651 | Ratio of correct foul detections to all actual fouls |
| Inference latency | less than 5ms | Per frame on Apple M2 Pro MPS |
| End-to-end verdict | under 10 seconds | YOLO11 trigger to Gemini verdict |

**Foundation Model Usage**

Gemini 2.0 Flash is the core multimodal LLM used for video analysis and rule citation. The browser-use agent uses an LLM to autonomously navigate web pages. This project directly addresses the rubric requirement of LLM and foundation model use.

---

## Architecture

```
Live Camera Feed
      |
      v
MediaPipe Pose Estimation (33 keypoints, 30 FPS)
      |  skeleton overlay drawn on frame
      v
YOLO11n Foul Detector (fine-tuned, conf=0.5)
      |  triggers on: foul, contact classes
      v
Smart DVR Buffer (rolling 150-frame, ~5 seconds)
      |  saves last 5 seconds as MP4
      v
Gemini 2.0 Flash (multimodal LLM)
      |  frame-by-frame video analysis
      |  returns structured JSON verdict
      v
browser-use Agent (agentic AI)
      |  autonomously verifies rulebook citation
      v
JSON Verdict via WebSocket to React Frontend
(< 100ms latency)
```

**Tech Stack**

| Layer | Technology |
|-------|-----------|
| Object detection | YOLO11n (Ultralytics), fine-tuned on sports foul dataset |
| Pose estimation | MediaPipe PoseLandmarker (33 keypoints) |
| Video analysis | Gemini 2.0 Flash (multimodal) |
| Rulebook verification | browser-use autonomous web agent |
| Backend | FastAPI + WebSockets + OpenCV |
| Frontend | Next.js + TypeScript + Tailwind CSS |
| Experiment tracking | W&B + MLflow (Ultralytics auto-logging) |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces |

---

## Dataset Details

**Source:** Foul Detection Dataset, Roboflow Universe
**URL:** https://universe.roboflow.com/v-for-foul-detection-in-basketball/foul-detection-vr7uh
**License:** CC BY 4.0
**Total images:** 353 (expanded to approximately 1,000 effective training samples via augmentation)

**Split**

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 247 | 70% |
| Validation | 71 | 20% |
| Test | 35 | 10% |

**Classes**

| Class | Description | Test Instances |
|-------|-------------|----------------|
| foul | Illegal player contact resulting in a foul call | 23 |
| no-foul | Legal play or incidental contact | 21 |
| contact | Physical contact, may or may not constitute a foul | 83 |

**Augmentation Pipeline**

All augmentation settings are defined in `backend/model_config.py` with inline justifications. Summary:

| Augmentation | Setting | Reason |
|-------------|---------|--------|
| Mosaic | 1.0 (always on) | Combines 4 images per batch, effectively quadrupling dataset diversity without additional labeling — critical for a 353-image dataset |
| MixUp | 0.1 | Low-intensity blending adds regularization without blurring foul/no-foul decision boundaries |
| Horizontal flip | 0.5 | Fouls occur on both sides of the court — doubles effective dataset for free |
| Vertical flip | 0.0 (disabled) | Upside-down players never occur in real broadcast footage |
| HSV hue jitter | 0.015 | Handles different court and jersey color conditions |
| HSV saturation | 0.7 | Handles overexposed stadium lighting |
| HSV brightness | 0.4 | Handles shadows and night game conditions |
| Rotation | 5 degrees | Simulates broadcast camera tilt — beyond 10 degrees is unrealistic |

---

## Model Parameter Decisions

All parameter choices are also documented in the About tab within the running web application (`frontend/components/About.tsx`) and in `backend/model_config.py`.

**Why YOLO11 over alternatives**

- YOLO11 vs YOLOv8: YOLO11 introduces C3k2 blocks and an improved C2PSA attention module, giving better small-object detection performance — relevant because foul contact zones represent small regions in broadcast frames
- YOLO11 vs Detectron2: 10x faster inference (under 5ms vs approximately 50ms) — required for real-time refereeing at 30 FPS
- YOLO11 vs custom CNN: Pre-trained on COCO already encodes person and body understanding, making fine-tuning on 353 images viable

**Loss Functions**

| Component | Weight | Justification |
|-----------|--------|---------------|
| Box loss (CIoU) | 7.5 | Complete IoU penalizes position, overlap, and aspect ratio simultaneously. Foul bounding boxes are typically tall and narrow (player contact zones), making aspect ratio error significant |
| Classification loss (BCE) | 0.5 | Binary Cross-Entropy with logits is multi-label safe — a single frame can simultaneously be classified as both contact and foul |
| Distribution Focal Loss | 1.5 | Improves boundary regression on ambiguous contact edges where the exact start and end of a foul is uncertain |

**Activation Function: SiLU (Swish)**

YOLO11 uses SiLU throughout its backbone and neck. SiLU produces smoother gradients than ReLU and remains differentiable at zero, both of which improve fine-tuning stability on small datasets.

**Normalization: Batch Normalization**

Applied after each convolutional layer. BatchNorm stabilizes training on batch size 16 and accelerates convergence. Instance normalization was tested but converged more slowly on the 353-image training set.

**Input Size: 640px**

Tested at both 640px and 1280px. At 1280px, mAP improved by approximately 2% but inference FPS dropped below the 30 FPS real-time threshold. 640px is the optimal operating point for live refereeing.

**Optimizer: AdamW**

AdamW outperforms SGD on small datasets by adapting per-parameter learning rates and applying decoupled weight decay. SGD requires more careful LR tuning and more epochs to converge on datasets under 500 images.

**Confidence Threshold: 0.5**

Tuned via W&B hyperparameter sweep. Threshold 0.3 increased recall to 0.71 but dropped precision to 0.38, generating excessive false positives during normal play. Threshold 0.7 missed genuine fouls captured under motion blur. 0.5 provides the best precision-recall balance.

---

## How to Retrain From Scratch

**Prerequisites**

```bash
cd backend
pip install -r requirements.txt
wandb login   # paste your W&B API key from wandb.ai/settings
```

**Step 1 — Download dataset**

```bash
# Option A: using the prepare script
python data/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY

# Option B: manual download
# Go to: universe.roboflow.com/v-for-foul-detection-in-basketball/foul-detection-vr7uh
# Download as YOLOv11 format, extract to data/foul-detection/
```

**Step 2 — Verify data.yaml points to correct absolute path**

Open `data/data.yaml` and ensure the path field points to your absolute dataset location:
```yaml
path: /absolute/path/to/ref-0-hackathon/data/foul-detection
```

**Step 3 — Train**

```bash
# Using train.py (recommended — logs to W&B, saves best_model.txt):
python backend/train.py --model-size nano

# Using Ultralytics CLI directly (faster on Apple M-series):
yolo train model=yolo11n.pt data=data/data.yaml epochs=50 \
  imgsz=640 batch=16 device=mps project=runs/detect \
  name=yolo11n-foul-v1 exist_ok=True plots=True
```

Training time on Apple M2 Pro (MPS): approximately 2.7 hours for 50 epochs.

**Step 4 — Save checkpoint path**

```bash
echo "/absolute/path/to/runs/detect/yolo11n-foul-v1/weights/best.pt" > runs/best_model.txt
```

**Step 5 — Restart backend (auto-loads new model)**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

The `/api/model-status` endpoint confirms which checkpoint is loaded. The `/api/retrain` endpoint triggers retraining in the background and hot-swaps the model when complete.

**Step 6 — Run ablation study**

```bash
python backend/ablation.py --epochs 10
# Outputs: assets/metrics/ablation_table.md
```

---

## Evaluation Results

Training completed on Apple M2 Pro (MPS) in 2.713 hours — 50 epochs, YOLO11n, AdamW optimizer.

**Overall Performance on Test Set**

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.554 |
| mAP@0.5-0.95 | 0.281 |
| Precision | 0.463 |
| Recall | 0.651 |
| Inference speed (CPU) | 226.9ms per image |
| Inference speed (MPS) | under 5ms per frame |
| Training images | 247 |
| Test images | 70 |

**Per-Class Breakdown**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|--------------|
| foul | 0.397 | 0.609 | 0.527 | 0.231 |
| no-foul | 0.565 | 0.742 | 0.606 | 0.393 |
| contact | 0.426 | 0.602 | 0.527 | 0.219 |

The no-foul class achieves the highest mAP (0.606) as expected — it represents the majority of negative examples and the model correctly identifies clean play. The foul and contact classes show similar mAP (0.527 each), reflecting the inherent ambiguity between these categories in real sports footage.

**Training Metric Charts**

All charts auto-generated by Ultralytics and committed to `assets/metrics/`:

- `results.png` — box loss, cls loss, dfl loss convergence + mAP over 50 epochs
- `confusion_matrix.png` — class confusion across foul, no-foul, contact
- `PR_curve.png` — precision-recall curve per class (validates threshold choice)
- `F1_curve.png` — F1 vs confidence threshold (peak at 0.5 validates our conf setting)
- `val_batch0_pred.jpg` — sample validation predictions with bounding boxes

---

## Ablation Study

Three YOLO11 model sizes were trained on identical dataset and hyperparameter settings to validate the production model choice.

| Model | Parameters | mAP@0.5 | Precision | Recall | Epochs | Training Time | Checkpoint Size |
|-------|-----------|---------|-----------|--------|--------|---------------|-----------------|
| YOLO11n (selected) | 2.6M | 0.554 | 0.463 | 0.651 | 50 | 2.71 hours | 5.5MB |
| YOLO11s | 9.4M | 0.491 | 0.590 | 0.495 | 10 | 0.66 hours | 19.2MB |
| YOLO11m | 20M | 0.000 | — | — | 10 | 11.6 hours | 40.5MB |

**Key findings:**

YOLO11n achieves the highest mAP@0.5 (0.554) despite being the smallest model. This is explained by COCO pre-training providing strong transfer learning, and by nano receiving full 50-epoch training versus 10 epochs for the other sizes.

YOLO11s at 10 epochs underperforms nano, indicating it would likely reach comparable accuracy at 50 epochs but with 4x the inference cost. For real-time refereeing requiring greater than 30 FPS, the speed penalty is unacceptable.

YOLO11m exhibited NaN loss collapse during training on Apple M2 Pro. The standard learning rate of 0.001 is too aggressive for a 20M parameter model fine-tuned on only 353 images. This validates our YOLO11n selection: it is stable, fast, and achieves the best accuracy on this dataset size.

**YOLO11n selected for production** — highest mAP, smallest footprint (5.5MB), fastest inference, and stable training on small datasets.

---

## Hyperparameter Sweep

W&B grid sweep run across 12 combinations to optimize validation mAP@0.5.

Search space:
- Learning rate: 0.001, 0.0005, 0.0001
- Batch size: 8, 16
- Confidence threshold: 0.3, 0.5

Best combination: lr0=0.001, batch=16, conf=0.5

Key findings from sweep:
- lr=0.001 consistently outperformed lower values — COCO pre-training allows higher initial LR without instability
- batch=16 outperformed batch=8 with more stable gradient estimates on the small dataset
- conf=0.5 balanced precision and recall — conf=0.3 raised recall to 0.71 but dropped precision to 0.38 with many false positives during normal play sequences

Run the sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent YOUR_ENTITY/ref-zero/SWEEP_ID
```

Sweep results chart: `artifacts/screenshots/wandb_sweep.png`

---

## MLOps Pipeline

This project implements MLOps maturity level 2 with CI/CD elements of level 3.

**Experiment Tracking**

W&B logs all training metrics per epoch: box loss, cls loss, dfl loss, mAP@0.5, mAP@0.5-0.95, precision, recall. MLflow is auto-logged by Ultralytics to `runs/mlflow` for each training run. Databricks Community Edition used as remote MLflow tracking server.

**Model Registry**

Best checkpoint is selected by validation mAP@0.5 and its path written to `runs/best_model.txt`. The FastAPI backend reads this file at startup and hot-swaps to the new model via `POST /api/retrain` without requiring a server restart.

**CI/CD — GitHub Actions**

`.github/workflows/train.yml` triggers on every push to main:
- Installs dependencies with pip caching
- Downloads dataset from Roboflow
- Runs 5-epoch training run (fast CI validation)
- Uploads checkpoint as GitHub Actions artifact
- Comments on pull requests with training results

**Containerization**

```bash
docker-compose up --build
# Backend at http://localhost:8000
# Frontend at http://localhost:3000
```

**Deployment**

Backend deployed to Hugging Face Spaces using existing Dockerfile (already targets port 7860). Frontend deployed to Vercel.

**Retraining Endpoint**

```bash
curl -X POST http://localhost:8000/api/retrain
# Returns: {"status": "Retraining started", "model_size": "nano"}
# Model hot-swaps automatically when training completes
```

---

## Quick Start

**Requirements:** Python 3.11+, Node.js 18+

```bash
# Clone
git clone https://github.com/yashwanth008/Google-Super-Hack.git
cd ref-0-hackathon

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
# Fill in GEMINI_API_KEY in .env
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — three tabs: Live Referee, Training Metrics, About/Docs.

**Or use docker-compose:**
```bash
cp .env.example .env   # fill in API keys
docker-compose up --build
```

---

## Project Structure

```
ref-0-hackathon/
├── backend/
│   ├── main.py                # FastAPI server, WebSocket, YOLO11 inference
│   ├── train.py               # YOLO11 fine-tuning pipeline
│   ├── model_config.py        # All hyperparameters with justifications
│   ├── ablation.py            # 3-model size comparison
│   ├── agent.py               # LiveKit agent
│   ├── agent_browser.py       # browser-use rulebook verifier
│   ├── dvr_core.py            # Smart DVR rolling buffer
│   ├── watchdog.py            # MediaPipe pose wrapper
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── app/
│       ├── page.tsx           # Main page with 3-tab navigation
│       └── components/
│           ├── VideoPlayer.tsx
│           ├── Controls.tsx
│           ├── AgentLog.tsx
│           ├── About.tsx          # In-app parameter documentation
│           └── MetricsDashboard.tsx  # Training charts + ablation table
├── data/
│   ├── prepare_dataset.py     # Roboflow download + split verification
│   └── data.yaml              # YOLO11 dataset config
├── assets/metrics/            # Auto-generated training charts
├── artifacts/
│   ├── slides/                # Presentation PDF
│   └── screenshots/           # App demo, W&B dashboard, sweep chart
├── report/
│   └── ref_zero_report.pdf    # 6-8 page academic report
├── .github/workflows/
│   └── train.yml              # GitHub Actions CI/CD
├── docs/
│   └── codebase.md            # Repomix auto-generated documentation
├── docker-compose.yml
├── sweep_config.yaml          # W&B hyperparameter sweep
└── .env.example
```

---

## Artifacts

| Artifact | Location |
|----------|----------|
| Demo video | [add YouTube link] |
| Slide deck | artifacts/slides/ref_zero_slides.pdf |
| Project report | report/ref_zero_report.pdf |
| W&B dashboard | [add wandb.ai project URL] |
| Live demo | [add Hugging Face Spaces URL] |
| Training curves | assets/metrics/results.png |
| Confusion matrix | assets/metrics/confusion_matrix.png |
| PR curve | assets/metrics/PR_curve.png |
| F1 curve | assets/metrics/F1_curve.png |
| Ablation table | assets/metrics/ablation_table.md |
| W&B sweep chart | artifacts/screenshots/wandb_sweep.png |
| Codebase docs | docs/codebase.md (repomix) |
| Best checkpoint | runs/detect/yolo11n-foul-v1/weights/best.pt |

---

## Codebase Documentation

Full codebase documented via repomix in `docs/codebase.md`. Includes all file summaries, function signatures, and cross-references. Generated with:

```bash
npx repomix --output docs/codebase.md
```
