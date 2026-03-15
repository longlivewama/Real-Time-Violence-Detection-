

---

# Real-Time Violence Detection System Using YOLOv8

## 1. Project Overview

The primary objective of this project is to develop a **real-time automated surveillance system capable of detecting violent interactions in environments such as schools, corridors, hospitals, and public areas**. The system utilizes the **YOLOv8 (You Only Look Once) deep learning architecture** to analyze video streams and distinguish between violent and non-violent human interactions.

The proposed system aims to address two fundamental challenges in automated surveillance:

**Low-Latency Processing**

The model must process live video streams quickly enough to enable **immediate detection and response** in real-world security systems.

**Detection of Small and Distant Actions**

Violent interactions may occur far from the camera. Therefore, the system processes frames at **higher spatial resolution** to preserve critical visual features.

By combining high-resolution visual input with an optimized deep learning architecture, the system provides an effective solution for **automated behavioral monitoring and public safety applications**.

---

# 2. Model Architecture and Technical Specifications

The system is built using the **YOLOv8n (Nano)** architecture, which is designed for high-speed inference while maintaining strong detection performance.

### Model Characteristics

| Property          | Value                 |
| ----------------- | --------------------- |
| Architecture      | YOLOv8n (Nano)        |
| Total Layers      | 130                   |
| Parameters        | 3,011,238             |
| GFLOPs            | ~8.2                  |
| Input Resolution  | 832 × 832             |
| Framework         | PyTorch + Ultralytics |
| Training Hardware | NVIDIA Tesla T4 GPU   |

The model achieves a balance between **computational efficiency and detection accuracy**, making it suitable for both **cloud deployment and edge devices**.

---

# 3. GPU Memory Management and VRAM Utilization

The system was trained and evaluated using an **NVIDIA Tesla T4 GPU**, which provides **14.9 GB of VRAM**.

### Parameter Storage

The model contains **approximately 3.01 million parameters** representing the learned weights and biases of the neural network.

Under standard **32-bit floating-point precision (FP32)**:

* Each parameter occupies **4 bytes**
* Total weight memory ≈ **12 MB**

However, the training pipeline utilizes **Automatic Mixed Precision (AMP)**, allowing the model to perform computations in **16-bit precision (FP16)** where applicable. This significantly reduces the memory footprint and increases computational throughput.

---

### Activation Maps

While model weights occupy relatively small memory, the majority of VRAM consumption originates from **activation maps generated during forward propagation**.

Given the input resolution:

**832 × 832 pixels**

Each convolutional layer produces feature maps that must be stored temporarily in memory for gradient computation during backpropagation.

These activation tensors are the primary reason GPU memory consumption increases during training.

---

###Tech Stack diagram

Camera → OpenCV → YOLOv8 Model → Detection Output
              ↓
           PyTorch
              ↓
        FastAPI API
              ↓
           Docker
              ↓
             AWS
---


### Computational Load

The model performs approximately:

**8.1 – 8.2 GFLOPs (Giga Floating Point Operations)**

This metric represents the total number of mathematical operations required for a single forward pass through the network.

Efficient GPU parallelization allows these operations to be executed within **milliseconds per frame**, enabling real-time processing.

---

### Layer Fusion Optimization

During inference, the architecture applies **layer fusion optimization**, reducing the effective layer count from:

**130 layers → 73 fused layers**

This process merges operations such as:

* Convolution
* Batch Normalization

into single computational kernels.

The advantages include:

* Reduced memory transfers
* Faster inference
* Lower computational overhead

---

# 4. Dataset Description [link](https://universe.roboflow.com/shah-xxxqs/violence-3h8pw)

The training dataset consists of **6,846 annotated images** representing various human interactions.

### Dataset Characteristics

| Property            | Value                                |
| ------------------- | ------------------------------------ |
| Total Images        | 6,846                                |
| Classes             | Violence / Non-Violence              |
| Environments        | Corridors, outdoor areas, classrooms |
| Lighting Conditions | Mixed (indoor & outdoor)             |

The dataset provides **significant environmental diversity**, which improves the model’s ability to generalize across different real-world conditions.

---

### Class Definition

The dataset focuses on **binary behavioral classification**:

| Class        | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| Violence     | Physical altercations, aggressive postures, combat interactions |
| Non-Violence | Normal social interactions and neutral behavior                 |

By limiting the classification space to **two behavioral classes**, the model learns to focus specifically on **human motion patterns and interaction geometry** rather than static object recognition.

---

# 5. Training Configuration

The training process was conducted using the **Ultralytics YOLOv8 framework** with the following configuration.

| Parameter        | Value  |
| ---------------- | ------ |
| Optimizer        | SGD    |
| Learning Rate    | 0.001  |
| Momentum         | 0.937  |
| Weight Decay     | 0.0005 |
| Epochs           | 35     |
| Batch Size       | 16     |
| Input Resolution | 832    |

The use of **Stochastic Gradient Descent (SGD)** ensures stable convergence while preventing excessive model overfitting.

---

# 6. System Performance Evaluation

The model performance was evaluated using standard computer vision metrics:

* **Precision**
* **Recall**
* **Mean Average Precision (mAP)**
* **F1 Score**

---

## Performance Results

| Class        | Precision | Recall | mAP50 | mAP50-95 |
| ------------ | --------- | ------ | ----- | -------- |
| All Classes  | 0.844     | 0.796  | 0.875 | 0.553    |
| Violence     | 0.882     | 0.836  | 0.910 | 0.570    |
| Non-Violence | 0.806     | 0.756  | 0.840 | 0.535    |

---

### Precision

Precision measures the system's ability to **avoid false alarms**.

A precision score of **0.844** indicates that the majority of predicted violent events correspond to actual violent incidents.

This is particularly important in surveillance systems where excessive false alerts can lead to **alarm fatigue**.

---

### Recall

Recall measures the system's ability to **detect all true violent events**.

The violence class achieved a recall score of **0.836**, indicating strong sensitivity in identifying real altercations.

---

### Mean Average Precision (mAP)

The **Violence class achieved a mAP50 score of 0.910**, demonstrating a high confidence level in identifying aggressive interactions.

This metric reflects the model's ability to accurately localize and classify violent behavior across different detection thresholds.

---

# 7. Real-Time Inference Performance

During deployment testing, the system achieved:

**3.6 milliseconds per frame inference time**

This translates to a theoretical throughput exceeding:

**270 Frames Per Second (FPS)**

Such performance allows the system to process live surveillance streams without perceptible delay.

---

# 8. Implementation Pipeline (MLOps)

The project includes a complete **machine learning pipeline**, covering:

### Data Engineering

A preprocessing script reorganizes the dataset into the **YOLO training structure**, ensuring efficient data loading and minimizing I/O bottlenecks during training.

---

### Automated Dataset Configuration

The training script dynamically generates a `data.yaml` file that maps dataset classes to model outputs. This prevents configuration errors and simplifies experimentation with different datasets.

---

### Persistent Backup System

The project integrates a backup mechanism that stores:

* Model weights (`best.pt`)
* Training metrics
* Evaluation plots

directly to **Google Drive**, ensuring results are preserved even when using ephemeral cloud environments such as Google Colab.

---

# 9. Conclusion

This project demonstrates that **high-performance violence detection systems can be achieved using lightweight deep learning architectures** when combined with high-resolution visual input and optimized GPU computation.


