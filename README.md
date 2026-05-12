# Ref Zero — AI-Powered Sports Referee Assistant

An end-to-end multimodal AI system that acts as a real-time Video Assistant Referee for sports. Ref Zero fine-tunes a YOLO11n object detection model on a curated sports foul dataset, streams live video through a FastAPI WebSocket backend with MediaPipe pose estimation, and triggers Gemini 2.0 Flash to analyze 5-second DVR clips when a foul is detected. Verdicts cite specific rulebook rules, which are autonomously verified by a browser-use agentic AI system.

**API Documentation:** https://ref-zero.onrender.com/docs  
**W&B Dashboard:** https://wandb.ai/youngtiger-paladugu-san-jose-state-university/ref-zero  
**Demo Video:** [YouTube Demo](https://drive.google.com/file/d/1_uCighf9juAedLwXcA47pu4ucPwO8M6a/view?usp=sharing)  
**Project Report:** [ref_zero_report.pdf](report/ref_zero_report.pdf)

**Slides:** [Slides](Slides/ref_zero_slides.pptx)

---

## Table of Contents

- [Team Contributions](#team-contributions)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Inputs, Outputs, and Key Metrics](#inputs-outputs-and-key-metrics)
- [System Architecture](#system-architecture)
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
- [Codebase Documentation](#codebase-documentation)

---

## Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| Yashwanth Paladugu | ML Lead and Backend | YOLO11 training pipeline, model_config.py, train.py, ablation.py, FastAPI WebSocket server, DVR buffer, Gemini 2.0 Flash integration, YOLO11 foul detection wiring in main.py |
| Zach Xie| Frontend | Next.js UI, VideoPlayer component, Controls, AgentLog, About tab, MetricsDashboard tab, tab navigation in page.tsx |
| Zach Xie | MLOps | Docker, docker-compose, GitHub Actions CI/CD workflow, Render deployment, W&B sweep configuration |
| Yashwanth Paladugu | Agents | browser-use rulebook verification agent, LiveKit multi-camera agent, agent_browser.py |

Each member's commits are visible in the repository history. Full codebase ownership is documented in [docs/codebase.md](docs/codebase.md), generated via repomix.

---

## Problem Statement

Every year, controversial referee calls cost teams championships, millions in revenue, and fan trust. The NFL employs over 450 officials across 256 games requiring split-second decisions under pressure. The NBA regularly sees missed calls swing playoff games. Soccer's VAR system exists but still depends on human interpretation. Youth sports — representing 90 percent of all games played — have no video review capability whatsoever.

Current video review systems require referees to stop the game for 2 to 5 minutes, manually scrub footage frame-by-frame, recall rules from 200-plus page rulebooks, and make judgment calls under intense pressure. Even with replay tools, human error remains significant.

Ref Zero addresses this by providing an AI assistant that detects fouls in real time, analyzes clips automatically, and returns structured verdicts with specific rulebook citations in under 10 seconds.

---

## Solution Overview

Ref Zero implements a five-stage pipeline:

1. Live camera feed is processed by MediaPipe at 30 FPS, tracking 33 body keypoints per player for real-time skeleton overlay
2. A fine-tuned YOLO11n model analyzes each frame — when foul or contact is detected above confidence 0.5, the DVR buffer is triggered
3. A rolling 150-frame buffer saves the last 5 seconds as an MP4 clip
4. Gemini 2.0 Flash receives the clip and returns a structured JSON verdict with sport, action breakdown, rule violated, verdict, and confidence score
5. A browser-use agentic AI autonomously opens the official rulebook URL and verifies the cited rule text

End-to-end review time: under 10 seconds, compared to 2 to 5 minutes for traditional VAR.

**Foundation Model:** Gemini 2.0 Flash (multimodal large language model) for natural language video analysis and rule citation

**Agentic AI:** browser-use agent that autonomously navigates web pages to verify official rulebook citations

---

## Inputs, Outputs, and Key Metrics

**Input**

- Live webcam stream via WebSocket (base64 encoded frames at 30 FPS)
- Uploaded MP4 video file via POST /api/trigger_review

**Output**

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

**Key Metrics**

| Metric | Value | Description |
|--------|-------|-------------|
| mAP@0.5 | 0.554 | Mean average precision at IoU threshold 0.5 |
| mAP@0.5-0.95 | 0.281 | Mean average precision across IoU thresholds |
| Precision | 0.463 | Ratio of correct detections to all detections |
| Recall | 0.651 | Ratio of correct detections to all actual fouls |
| Inference latency | less than 5ms | Per frame on Apple M2 Pro MPS |
| End-to-end verdict | under 10 seconds | YOLO11 trigger to Gemini rulebook verdict |
| Training time | 2.713 hours | 50 epochs on Apple M2 Pro MPS |

---

## System Architecture

```
Live Camera Feed
      |
      v
MediaPipe Pose Estimation  (33 keypoints, 30 FPS, skeleton overlay)
      |
      v
YOLO11n Foul Detector  (fine-tuned, conf=0.5, triggers DVR)
      |
      v
Smart DVR Buffer  (rolling 150 frames = 5 seconds, saved as MP4)
      |
      v
Gemini 2.0 Flash  (multimodal LLM, frame-by-frame analysis, JSON verdict)
      |
      v
browser-use Agent  (agentic AI, autonomous rulebook web verification)
      |
      v
WebSocket to Next.js Frontend  (< 100ms latency)
```

**Technology Stack**

| Layer | Technology |
|-------|-----------|
| Object detection | YOLO11n (Ultralytics), fine-tuned on sports foul dataset |
| Pose estimation | MediaPipe PoseLandmarker (33 keypoints) |
| Video analysis | Gemini 2.0 Flash (multimodal LLM) |
| Rulebook verification | browser-use autonomous web agent |
| Backend | FastAPI, WebSockets, OpenCV, Python 3.11 |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Experiment tracking | W&B, MLflow (auto-logged by Ultralytics) |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Deployment | Render (backend), Vercel (frontend) |

---

## Dataset Details

**Source:** Foul Detection Dataset, Roboflow Universe  
**URL:** https://universe.roboflow.com/v-for-foul-detection-in-basketball/foul-detection-vr7uh  
**License:** CC BY 4.0  
**Total images:** 353  
**Effective training samples:** approximately 1,000 (via 3x augmentation)

**Dataset Split**

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
| contact | Physical contact, may or may not be a foul | 83 |

**Augmentation Pipeline**

All settings defined in `backend/model_config.py` with inline justifications.

| Augmentation | Setting | Reason |
|-------------|---------|--------|
| Mosaic | 1.0 (always on) | Combines 4 images per batch, quadrupling effective diversity without extra labeling — critical for a 353-image dataset |
| MixUp | 0.1 | Adds regularization without blurring foul/no-foul decision boundaries |
| Horizontal flip | 0.5 | Fouls occur on both sides of the court — doubles effective dataset for free |
| Vertical flip | 0.0 (disabled) | Upside-down players never occur in real broadcast footage |
| HSV hue jitter | 0.015 | Handles different court and jersey color conditions |
| HSV saturation | 0.7 | Handles overexposed stadium lighting |
| HSV brightness | 0.4 | Handles shadows and night game conditions |
| Rotation | 5 degrees | Simulates broadcast camera tilt — beyond 10 degrees is unrealistic |

---

## Model Parameter Decisions

All parameter choices are documented in `backend/model_config.py` and visible in the About tab of the running web application.

**Why YOLO11 over alternatives**

- YOLO11 vs YOLOv8: C3k2 blocks and improved C2PSA attention provide better small-object detection for contact zones between players
- YOLO11 vs Detectron2: 10x faster inference (under 5ms vs approximately 50ms) — required for real-time refereeing at 30 FPS
- YOLO11 vs custom CNN: COCO pre-training encodes person and body understanding, making fine-tuning on 353 images viable

**Loss Functions**

| Component | Weight | Justification |
|-----------|--------|---------------|
| Box loss (CIoU) | 7.5 | Complete IoU penalizes position, overlap, AND aspect ratio. Foul bounding boxes are tall and narrow — aspect ratio error matters |
| Classification loss (BCE) | 0.5 | Binary Cross-Entropy with logits is multi-label safe — a frame can simultaneously be contact AND foul |
| Distribution Focal Loss | 1.5 | Improves boundary regression on ambiguous contact edges where exact foul start and end is uncertain |

**Activation Function: SiLU**

YOLO11 uses SiLU throughout. SiLU produces smoother gradients than ReLU and remains differentiable at zero, improving fine-tuning stability on small datasets.

**Normalization: Batch Normalization**

Applied after each convolutional layer. BatchNorm stabilizes training on batch size 16 and accelerates convergence compared to InstanceNorm on small datasets.

**Input Size: 640px**

Tested at both 640px and 1280px. At 1280px, mAP improved by approximately 2% but inference FPS dropped below the 30 FPS real-time threshold. 640px is the optimal operating point.

**Optimizer: AdamW**

AdamW outperforms SGD on small datasets via adaptive per-parameter learning rates and decoupled weight decay. SGD requires significantly more epochs to converge on datasets under 500 images.

**Confidence Threshold: 0.5**

Tuned via W&B hyperparameter sweep. Threshold 0.3 increased recall to 0.71 but dropped precision to 0.38 with excessive false positives. Threshold 0.7 missed genuine fouls under motion blur. 0.5 provides the best precision-recall balance as validated by the F1 curve.

---

## How to Retrain From Scratch

**Prerequisites**

```bash
cd backend
pip install -r requirements.txt
wandb login
```

**Step 1 — Download dataset**

```bash
python data/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY

# Or manually: download from universe.roboflow.com/v-for-foul-detection-in-basketball/foul-detection-vr7uh
# Extract to data/foul-detection/
```

**Step 2 — Verify data.yaml uses absolute path**

```yaml
path: /absolute/path/to/data/foul-detection
train: train/images
val:   valid/images
test:  test/images
nc: 3
names: ['foul', 'no-foul', 'contact']
```

**Step 3 — Train**

```bash
# Using train.py (recommended — logs to W&B, saves checkpoint path):
python backend/train.py --model-size nano

# Using Ultralytics CLI directly (faster on Apple M-series with MPS):
yolo train model=yolo11n.pt data=data/data.yaml epochs=50 \
  imgsz=640 batch=16 device=mps project=runs/detect \
  name=yolo11n-foul-v1 exist_ok=True plots=True
```

**Step 4 — Save checkpoint path**

```bash
echo "/absolute/path/to/runs/detect/yolo11n-foul-v1/weights/best.pt" > runs/best_model.txt
```

**Step 5 — Restart backend**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
# Backend auto-loads new checkpoint from runs/best_model.txt
```

**Step 6 — Run ablation study**

```bash
python backend/ablation.py --epochs 10
# Outputs: assets/metrics/ablation_table.md
```

**Trigger retraining via API (no restart needed):**

```bash
curl -X POST https://ref-zero.onrender.com/api/retrain
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
| Parameters | 2,582,737 |
| GFLOPs | 6.3 |

**Per-Class Breakdown**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|--------------|
| foul | 0.397 | 0.609 | 0.527 | 0.231 |
| no-foul | 0.565 | 0.742 | 0.606 | 0.393 |
| contact | 0.426 | 0.602 | 0.527 | 0.219 |

The no-foul class achieves the highest mAP (0.606) as expected — it represents the majority of negative examples and the model correctly identifies clean play. The foul and contact classes show similar mAP (0.527), reflecting inherent ambiguity between categories in real sports footage.

**Training Metric Charts**

All charts auto-generated by Ultralytics, committed to `assets/metrics/`:

| File | Description |
|------|-------------|
| results.png | Box loss, cls loss, dfl loss convergence and mAP over 50 epochs |
| confusion_matrix.png | Class confusion across foul, no-foul, contact |
| PR_curve.png | Precision-recall curve per class (validates threshold choice) |
| F1_curve.png | F1 score vs confidence threshold (peak at 0.5 validates conf setting) |
| val_batch0_pred.jpg | Sample validation predictions with bounding boxes |

---

## Ablation Study

Three YOLO11 model sizes trained on identical dataset and hyperparameters to validate production model choice.

| Model | Parameters | mAP@0.5 | Precision | Recall | Epochs | Training Time | Checkpoint Size |
|-------|-----------|---------|-----------|--------|--------|---------------|-----------------|
| YOLO11n (selected) | 2.6M | 0.554 | 0.463 | 0.651 | 50 | 2.71 hours | 5.5MB |
| YOLO11s | 9.4M | 0.491 | 0.590 | 0.495 | 10 | 0.66 hours | 19.2MB |
| YOLO11m | 20M | 0.000 | — | — | 10 | 11.6 hours | 40.5MB |

**Key Findings**

YOLO11n achieves the highest mAP@0.5 despite being the smallest model. COCO pre-training provides strong transfer learning, and nano received full 50-epoch training versus 10 epochs for the other sizes.

YOLO11s at 10 epochs underperforms nano, indicating it would likely reach comparable accuracy at 50 epochs but at 4x the inference cost and size.

YOLO11m exhibited NaN loss collapse. The standard learning rate of 0.001 is too aggressive for a 20M parameter model fine-tuned on only 353 images. This validates YOLO11n selection: it is stable, fast, and achieves the best accuracy on this dataset size. Real-time refereeing requires greater than 30 FPS — only nano meets this constraint.

---

## Hyperparameter Sweep

W&B comparison of 3 learning rates to optimize validation mAP@0.5. Results visible in the W&B dashboard.

**Search space:**
- Learning rate: 0.001, 0.0005, 0.0001
- Batch size: 8, 16
- Confidence threshold: 0.3, 0.5

**Best combination:** lr0=0.001, batch=16, conf=0.5

**Findings:**
- lr=0.001 consistently best — COCO pre-training allows higher initial LR without instability
- batch=16 outperforms batch=8 with more stable gradient estimates on the small dataset
- conf=0.5 balanced precision and recall — conf=0.3 raised recall but caused excessive false positives during normal play

Sweep screenshot: [artifacts/screenshots/wandb_sweep.png](artifacts/screenshots/wandb_sweep.png)

---

## MLOps Pipeline

This project implements MLOps maturity level 2 with CI/CD elements.

**Experiment Tracking**

W&B logs all training metrics per epoch: box loss, cls loss, dfl loss, mAP@0.5, mAP@0.5-0.95, precision, recall. MLflow is auto-logged by Ultralytics to `runs/mlflow` for every training run.

**Model Registry**

Best checkpoint selected by validation mAP@0.5. Path written to `runs/best_model.txt`. FastAPI backend reads this file at startup and hot-swaps to the new model via `POST /api/retrain` without requiring a server restart.

**GitHub Actions CI/CD**

`.github/workflows/train.yml` triggers on every push to main:
- Installs dependencies with pip caching
- Creates minimal smoke-test dataset
- Runs 2-epoch YOLO11 training to verify pipeline integrity
- Uploads checkpoint as GitHub Actions artifact

**Containerization**

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

**Deployment**

Backend deployed to Render (https://ref-zero.onrender.com). Frontend deployed to Vercel. Both trigger automatic redeploy on push to main.

**Retraining Endpoint**

```bash
curl -X POST https://ref-zero.onrender.com/api/retrain
# Triggers YOLO11 retraining in background
# Auto-reloads new checkpoint when training completes

curl https://ref-zero.onrender.com/api/model-status
# Returns current checkpoint and training status
```

---

## Quick Start

**Requirements:** Python 3.11+, Node.js 18+

```bash
git clone https://github.com/yashwanth008/CMPE-258-Final-Project-Ref0.git
cd CMPE-258-Final-Project-Ref0

cp .env.example .env
# Fill in GEMINI_API_KEY in .env
```

**Backend:**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — three tabs: Live Referee, Training Metrics, About/Docs.

**Or use docker-compose:**

```bash
docker-compose up --build
```

**Train the model:**

```bash
python data/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY
python backend/train.py --model-size nano
```

---

## Project Structure

```
CMPE-258-Final-Project-Ref0/
├── backend/
│   ├── main.py              FastAPI server, WebSocket handler, YOLO11 inference
│   ├── train.py             YOLO11 fine-tuning pipeline with W&B logging
│   ├── model_config.py      All hyperparameters with justifications (100+ lines)
│   ├── ablation.py          3-model size comparison study
│   ├── agent.py             LiveKit multi-camera agent
│   ├── agent_browser.py     browser-use rulebook verification agent
│   ├── dvr_core.py          Smart DVR rolling buffer (150 frames = 5 seconds)
│   ├── watchdog.py          MediaPipe pose estimation wrapper
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── app/
│       ├── page.tsx         Main page with 3-tab navigation
│       └── components/
│           ├── VideoPlayer.tsx
│           ├── Controls.tsx
│           ├── AgentLog.tsx
│           ├── About.tsx          In-app parameter documentation (required by rubric)
│           └── MetricsDashboard.tsx  Training charts and ablation table
├── data/
│   ├── prepare_dataset.py   Roboflow download and split verification
│   └── data.yaml            YOLO11 dataset configuration
├── assets/
│   └── metrics/             Auto-generated training charts committed here
│       ├── results.png
│       ├── confusion_matrix.png
│       ├── PR_curve.png
│       └── F1_curve.png
├── artifacts/
│   ├── slides/              Presentation PDF
│   └── screenshots/         App demo, W&B dashboard, sweep chart, CI screenshot
├── report/
│   └── ref_zero_report.pdf  6-page academic report with all required sections
├── docs/
│   └── codebase.md          Repomix auto-generated codebase documentation
├── .github/workflows/
│   └── train.yml            GitHub Actions CI/CD training workflow
├── docker-compose.yml
├── sweep_config.yaml        W&B hyperparameter sweep definition
└── .env.example
```

---

## Artifacts

| Artifact | Location / Link |
|----------|----------------|
| Live backend | https://ref-zero.onrender.com |
| API documentation | https://ref-zero.onrender.com/docs |
| W&B dashboard | https://wandb.ai/youngtiger-paladugu-san-jose-state-university/ref-zero |
| Demo video | [Add YouTube link] |
| Slide deck | artifacts/slides/ref_zero_slides.pdf |
| Project report | report/ref_zero_report.pdf |
| Training curves | assets/metrics/results.png |
| Confusion matrix | assets/metrics/confusion_matrix.png |
| PR curve | assets/metrics/PR_curve.png |
| Ablation table | assets/metrics/ablation_table.md |
| W&B sweep screenshot | artifacts/screenshots/wandb_sweep.png |
| GitHub Actions screenshot | artifacts/screenshots/github_actions.png |
| Codebase documentation | docs/codebase.md |
| Best checkpoint | runs/detect/yolo11n-foul-v1/weights/best.pt (local only) |

---

## Codebase Documentation

Full codebase documented via repomix in [docs/codebase.md](docs/codebase.md). Includes all file summaries, function signatures, and cross-references.

Generated with:

```bash
npx repomix --output docs/codebase.md
```

---

## References

1. Redmon, J. et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR.
2. Cao, Z. et al. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation. IEEE TPAMI.
3. Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172.
4. Ultralytics. (2024). YOLO11. https://docs.ultralytics.com/models/yolo11/
5. Google DeepMind. (2024). Gemini 2.0 Flash. https://deepmind.google/technologies/gemini/
6. Roboflow. (2025). Foul Detection Dataset. https://universe.roboflow.com/v-for-foul-detection-in-basketball/foul-detection-vr7uh
7. browser-use. (2024). Web Automation for AI Agents. https://github.com/browser-use/browser-use