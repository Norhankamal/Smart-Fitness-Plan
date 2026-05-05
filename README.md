# FitPredict — Intelligent Fitness Goal & Calorie Optimization System

> A sequential machine learning pipeline that predicts user fitness goals, estimates daily caloric targets, and delivers fully personalized workout and nutrition plans through an interactive Flask web application.

---

## Overview

FitPredict operates as a three-stage connected ML pipeline where each model's output serves as input to the next, ensuring coherent and individualized recommendations across all stages.

| Stage | Task | Approach | Output |
|---|---|---|---|
| 1 | Fitness Goal Classification | Supervised Classification | Weight Loss · Muscle Gain · General Fitness |
| 2 | Calorie Burn Estimation | Regression Modeling | Daily caloric burn target |
| 3 | Plan Generation | Optimization Engine | Workout plan · Training schedule · Hydration targets |

---

## Pipeline Architecture

```
User Profile Input
(Age, Weight, Height, Activity Level, Body Metrics ...)
          │
          ▼
 ┌─────────────────────────┐
 │  Stage 1                │
 │  Goal Classifier        │──▶  Fitness Goal
 └─────────────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │  Stage 2                │
 │  Calorie Predictor      │──▶  Daily Calorie Target
 └─────────────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │  Stage 3                │
 │  Plan Generator         │──▶  Personalized Fitness Plan
 └─────────────────────────┘     · Workout type & duration
                                  · Daily training hours
                                  · Water intake recommendation
```

---

## Technology Stack

```
Machine Learning     Scikit-learn
Web Framework        Flask
Data Processing      NumPy · Pandas
Visualization        Matplotlib · Seaborn
Language             Python 3.9+
```

---

## Project Structure

```
FitPredict/
│
├── app.py                        # Flask application — full pipeline integration
├── requirements.txt              # Project dependencies
│
├── functionsHM.ipynb             # Core ML logic — classification, prediction, plan generation
├── finalHMbeforeflask.ipynb      # Model training, validation, and evaluation
│
├── static/                       # CSS, JavaScript, and image assets
└── templates/                    # HTML templates for the web interface
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/FitPredict.git
cd FitPredict

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
python app.py

# 4. Access the interface
http://127.0.0.1:5000
```

---

## Future Work

- Wearable device integration for real-time biometric data input
- Interactive progress dashboard with visual analytics and historical tracking
- Cloud-based API deployment for scalable access
- AI-assisted conversational interface for dynamic plan adjustment

---

## Author

**Norhan Kamal Hosny** — AI / ML Engineer

Machine Learning · Predictive Modeling · Intelligent Recommendation Systems
