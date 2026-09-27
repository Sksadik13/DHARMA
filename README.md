# DHARMA
DHARMA is an AI-based autonomous driving framework for unstructured Indian roads. It combines YOLO, ByteTrack, drivable-area detection, motion prediction, collision-risk assessment, and adaptive path planning to detect hazards, predict movement, avoid collisions, and continuously select safer driving paths.
# DHARMA 🚗

### Dynamic Hazard-aware Autonomous Routing and Motion Adaptation

> **AI-based adaptive path planning and collision avoidance for autonomous vehicles on unstructured Indian roads.**

DHARMA is an AI-driven autonomous vehicle decision-making framework designed to handle complex and dynamically changing road environments. The system combines object detection, object tracking, road/drivable-area understanding, motion prediction, collision-risk assessment, and adaptive path planning into a continuous perception-to-planning pipeline.

The project was developed for **Smart India Hackathon (SIH) 2026** under the problem statement:

**SIH26037 — Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads**

---

## 📌 Problem Statement

Autonomous driving systems often assume structured road environments with clear lane markings, predictable traffic behavior, and well-defined road boundaries.

Indian roads can be significantly more challenging due to:

- Unclear or missing lane markings
- Mixed traffic
- Two-wheelers
- Pedestrians
- Animals
- Parked vehicles
- Narrow roads
- Irregular road boundaries
- Unexpected obstacles
- Dynamically changing traffic conditions

A vehicle operating in such an environment cannot rely only on predefined lanes or a fixed trajectory.

DHARMA addresses this problem by continuously observing the environment, understanding the available driving area, predicting the movement of surrounding objects, assessing collision risk, and adapting the vehicle's trajectory accordingly.

---

# 🎯 Objectives

The main objectives of DHARMA are:

- Detect and track surrounding road users.
- Understand the available drivable road area.
- Predict the future movement of detected objects.
- Estimate potential collision risks.
- Calculate and use Time-to-Collision (TTC).
- Generate multiple candidate driving trajectories.
- Select a safer trajectory based on the current environment.
- Adapt vehicle speed according to risk.
- Perform braking or emergency braking when required.
- Continuously replan the vehicle trajectory as the environment changes.
- Provide a simulation environment for controlled testing and evaluation.

---

# 🧠 DHARMA Pipeline

DHARMA follows a continuous perception-to-planning pipeline:

```text
                 CAMERA / VIDEO
                       │
                       ▼
              ┌─────────────────┐
              │ P1: PERCEPTION  │
              │ YOLO + ByteTrack│
              └────────┬────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │ P2: ROAD UNDERSTANDING│
           │   Drivable Area      │
           └───────────┬──────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │ P3: MOTION PREDICTION│
           │ Future Trajectories  │
           └───────────┬──────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │ P4: SAFETY & RISK    │
           │ Collision Assessment │
           └───────────┬──────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │ P5: PATH PLANNING    │
           │ Adaptive Trajectory  │
           └───────────┬──────────┘
                       │
                       ▼
                 VEHICLE ACTION
                       │
                       ▼
                    REPLAN
                       │
                       └───────────────►
