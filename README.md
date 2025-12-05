# DSMRF: A Dual-Stream Masked-Recurrent Framework for Ego-Centric Vehicular Crash Prediction

**[Submitted to IEEE VTC2026-Spring]**

This repository contains the official PyTorch implementation of the paper: **"DSMRF: A Dual-Stream Masked-Recurrent Framework for Ego-Centric Collision Anticipation in Real-World Vehicular Traffic"**.

Our framework achieves State-of-the-Art (SOTA) performance on the **Nexar Dashcam Collision Prediction Benchmark** by synergizing the fine-grained motion attention of **VideoMAE** with the temporal trajectory modeling of **CNN-BiGRU** networks.

---

## 🏎️ Abstract

Reliable online collision anticipation from monocular dashcam footage is a critical component of ADAS. Existing approaches often struggle to capture long-term temporal dependencies or suffer from mode collapse on imbalanced data. We propose **DSMRF**, a hybrid architecture that processes video streams through two complementary branches:
1.  **Masked Video Attention Stream:** Uses a pre-trained VideoMAE backbone to capture subtle motion dynamics and appearance anomalies.
2.  **Recurrent Spatial Stream:** Uses a ResNet-18 + BiGRU to model high-level object trajectories and scene context.

**Result:** DSMRF achieves a **mAP of 0.8977** on the challenging Nexar dataset, significantly outperforming 3D-CNN and pure Transformer baselines.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2Fe25107a6611b102ac587c07fcf39948f%2Fchrome_54w946rFmF.png?generation=1764946123496837&alt=media)
---

## 📊 Visual Results

The following figures are generated automatically during the training and evaluation of the model.

### 1. Training Dynamics
We employ a Cosine Annealing scheduler and Early Stopping based on validation mAP. The plots below show the convergence of Loss and the improvement of Mean Average Precision (mAP) over epochs.

![Training Dynamics](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2Fb5a62245c381d95d27f2b512bd997406%2Floss.png?generation=1764945714030971&alt=media)
*(Left: Training vs Validation Loss. Right: Validation mAP across horizons)*

### 2. ROC Curves (Test Set)
Performance evaluation across three anticipation horizons (**0.5s**, **1.0s**, **1.5s**). The high AUC scores indicate robust separation between collision and non-collision events even 1.5 seconds before impact.

![ROC Curves](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2Faac727880ba6dfd6bc2ada66cfb081bb%2Froc.png?generation=1764945834003912&alt=media)

### 3. Confusion Matrices
Detailed breakdown of true positives and false alarms at different time-to-collision (TTC) horizons.

| Horizon 0.5s | Horizon 1.0s | Horizon 1.5s |
| :---: | :---: | :---: |
| ![CM 0.5s](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2F0e369f140a50a5bb417804b9ae7d69cc%2F0.5.png?generation=1764945859835341&alt=media) | ![CM 1.0s](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2F11624928cd589f56e022c1c62bc9ad67%2F1.0.png?generation=1764945895081212&alt=media) | ![CM 1.5s](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F19186184%2Fcd0ae973848497d8ddfdaadf70c65eb4%2F1.5.png?generation=1764945920728505&alt=media) |


## 📂 Dataset Preparation

We utilize the **Nexar Dashcam Collision Prediction Dataset**.
1.  Download the dataset from [Kaggle](https://www.kaggle.com/competitions/nexar-collision-prediction)
2.  Organize your data directory as follows:

```text
/data
  ├── train.csv         # Contains columns: id, target, time_of_event
  └── train_resized/    # Folder containing .mp4 video files
      ├── 00001_256x256.mp4
      ├── 00002_256x256.mp4
      └── ...
```

**Note:** The code expects videos resized to `256x256` or similar to speed up loading, though the model input is `224x224`.

## 🚀 Usage

### Configuration
Modify the `CONFIG` section in `main.py` to match your paths:
```python
CSV_PATH = "/path/to/your/train.csv"
TRAIN_VIDEO_DIR = "/path/to/your/videos"
OUTPUT_DIR = "./nexar_hybrid_outputs"
BATCH_SIZE = 4  # Adjust based on VRAM 
```

### Training
Run the main script to train the model, evaluate on the test set, and generate all plots:

```bash
DSMRF.ipynb [Link](https://github.com/borhanitrash/DSMRF/blob/main/DSMRF.ipynb)
```

## 📈 Performance Summary

Comparison against baselines on the Nexar Test Set (Held-out 20% split):

| Model | mAP | AP @ 0.5s | AP @ 1.0s | AP @ 1.5s | ROC-AUC (Avg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet-18 3D | 0.4917 | 0.5015 | 0.4905 | 0.4831 | 0.4980 |
| VideoMAE | 0.5118 | 0.5127 | 0.5111 | 0.5116 | 0.5244 |
| CNN-GRU | 0.5209 | 0.5334 | 0.5192 | 0.5101 | 0.5367 |
| **DSMRF (Ours)** | **0.8977** | **0.9438** | **0.9040** | **0.8452** | **0.8984** |


## ⚖️ License

This project is licensed under the MIT License.

## 🙏 Acknowledgement
We acknowledge the use of large-scale pre-trained models from [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [Torchvision](https://pytorch.org/vision).
