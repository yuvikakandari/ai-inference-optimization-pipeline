
# AI Inference Optimization & Performance Profiling Pipeline

An automated MLOps optimization and performance profiling pipeline that exports, simplifies, compresses, and benchmarks Deep Learning models for resource-constrained Edge-AI deployment.


## Table of Contents

* [Project Overview](https://www.google.com/search?q=%231-project-overview)
* [Architecture & Execution Flow](https://www.google.com/search?q=%232-architecture--execution-flow)
* [Tech Stack](https://www.google.com/search?q=%233-tech-stack)
* [How To Run](https://www.google.com/search?q=%234-how-to-run)
* [Design Decisions](https://www.google.com/search?q=%235-design-decisions)
* [Hardware Performance Bottlenecks](https://www.google.com/search?q=%236-hardware-performance-bottlenecks)
* [Evaluation & Empirical Results](https://www.google.com/search?q=%237-evaluation--empirical-results)
* [Known Limitations](https://www.google.com/search?q=%238-known-limitations)
* [Production Improvements](https://www.google.com/search?q=%239-production-improvements)

---

## 1. Project Overview

This repository provides an automated, end-to-end deployment pipeline designed to transition deep learning architectures from training frameworks to optimized edge execution engines. The pipeline automates the graph compilation process: exporting PyTorch models to Intermediate Representation formats (**ONNX**), running static structural graph graph-simplifications, compressing precision footprints via **Post-Training Static Quantization (PTQ)**, and profiling execution latency and throughput metrics directly on target hardware backends (**CPU** and **NVIDIA CUDA GPU**).

The case study leverages a pre-trained **MobileNetV2** architecture to evaluate downstream edge performance characteristics, compression ratios, and cross-hardware runtime dynamics.

---

## 2. Architecture & Execution Flow

```
   PyTorch Framework (MobileNetV2 FP32)
                 ↓
   [export.py]  → Compiles graph to ONNX Intermediate Representation
                 ↓
     models/onnx/mobilenet_v2.onnx (Raw Graph)
                 ↓
   [optimise.py] → Executes ONNX Simplifier (Constant Folding / Node Elimination)
                 ↓
     models/onnx/mobilenet_v2_simplified.onnx (Optimized FP32 Graph)
                 ↓
   [quantize.py] → Post-Training Static Quantization (Calibration Data Reader)
                 ↓
     models/onnx/mobilenet_v2_int8.onnx (Quantize-Dequantize [QDQ] Format)
                 ↓
   [benchmark.py] → Profiles Latency, Throughput (FPS), & Footprint Reductions
                 ↓
  Target Execution Providers: [CUDAExecutionProvider / CPUExecutionProvider]

```

### End-to-End Pipeline Execution:

1. **Graph Compilation:** The model graph is exported with clean operator set mappings (`opset_version=17`) and structural dynamic axes constraints.
2. **Graph Optimization:** A static pass eliminates redundant operations, pre-calculates fixed tensors through constant folding, and dissolves dead execution branches.
3. **Static Calibration:** A calibration data reader streams sample inputs to compute activation tensor scales and zero-point parameters.
4. **Quantization Compilation:** The pipeline outputs an optimized **QDQ (Quantize-Dequantize)** graph structure, reducing precision layers from 32-bit Floating Point down to signed 8-bit Integers.
5. **Hardware Profiling:** High-resolution hardware timers benchmark the compilation models under warm-up conditions, reporting execution latency and frame throughput metrics.

---

## 3. Tech Stack

| Component | Choice | Reason |
| --- | --- | --- |
| **Core Framework** | PyTorch 2.1+ | Standard deep learning framework for training and model definition layers. |
| **Deployment Format** | ONNX (Open Neural Network Exchange) | Interoperable, hardware-agnostic intermediate representation graph standard. |
| **Graph Optimizer** | ONNX Simplifier (`onnxsim`) | Performs symbolic structural checks; removes redundant identity nodes and folds constant tensors. |
| **Quantization Suite** | ONNX Runtime Quantization Tools | Built-in quantization toolkit supporting static calibration and data reader abstractions. |
| **Inference Engines** | ONNX Runtime (`onnxruntime-gpu`) | High-performance inference engine supporting multiple pluggable Execution Providers (EPs). |
| **Hardware Backends** | NVIDIA CUDA + CPU Execution | Direct mapping to native silicon execution pipelines (**NVIDIA RTX 2050 Tensor Cores** / x64 Host CPU). |
| **Image Processing** | OpenCV (`opencv-python`) | Production-grade matrix preprocessing wrapper (resizing, colorspace adjustments, transpose configurations). |

---

## 4. How To Run

### Prerequisites

* Python 3.10+
* Windows 10/11 or Linux Host
* NVIDIA Dedicated GPU (e.g., RTX 2050) with CUDA Toolkit 12.x and cuDNN 9.x installed on the host system system paths.

### Step 1 — Clone and Environment Setup

```powershell
git clone <your-repo-url>
cd EdgeAI-Inference-Engine
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

```

### Step 2 — Configure GPU Environment Bindings

To align Python variables with local NVIDIA drivers, make sure the required wheels are installed within your local environment:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

```

### Step 3 — Run the Optimization and Deployment Pipeline

Execute the optimization layers sequentially:

```powershell
# 1. Export PyTorch weights to raw ONNX representation
python scripts/export.py

# 2. Run graph simplification and constant folding passes
python scripts/optimise.py

# 3. Apply Post-Training Static Quantization (QDQ generation)
python scripts/quantize.py

# 4. Profile the performance metrics across active hardware runtimes
python scripts/benchmark.py

```

---

## 5. Design Decisions

### 5.1 Clean Architecture Graph Exporting

* **The Decision:** Standardizing the export pass using `opset_version=17` with explicitly declared `dynamic_axes` definitions mapping the batch layout.
* **Why:** Older operator sets lack support for advanced layers, causing graph generation failures. Opset 17 ensures robust mathematical representation mapping. Explicitly declaring `dynamic_axes={'input': {0: 'batch_size'}}` prevents the compiler from hardcoding static memory layouts, allowing production wrappers to scale dynamically from single-frame real-time webcam inference up to batch processing pipelines.

### 5.2 Post-Training Static Quantization (PTQ) over Dynamic Quantization

* **The Decision:** Transitioning from dynamic runtime scale calculations to **Post-Training Static Quantization** using a custom `CalibrationDataReader`.
* **Why:** Dynamic quantization computes scalar tensor values on-the-fly during active inference execution loops. For convolutional neural networks like MobileNetV2, this introduces massive computation overhead. Static PTQ evaluates activation scaling ranges beforehand using representative calibration frames, eliminating dynamic calculation bottlenecks entirely at runtime.

### 5.3 signed Int8 Matrix Mapping via QuantFormat.QDQ

* **The Decision:** Utilizing `QuantFormat.QDQ` with signed `QInt8` datatypes rather than the legacy `QOperator` format.
* **Why:** Legacy `QOperator` blocks alter the core structure of individual graph operations, which limits acceleration capabilities on modern graphics cards. The newer `QDQ` format inserts explicit **Quantize** and **Dequantize** boundary nodes directly into the computational graph. This structural layout allows the inference engine to optimize execution paths, leveraging hardware acceleration features like **NVIDIA Tensor Cores** or x64 CPU matrix extensions.

---

## 6. Hardware Performance Bottlenecks

During performance profiling on target hardware accelerators, an interesting behavioral trade-off was isolated: **the quantized INT8 model executed slower on the GPU than its baseline FP32 counterpart ($4.40\text{ ms}$ vs $3.09\text{ ms}$)**.

### The Root Cause: Memory Copy (`Memcpy`) Overhead

When compiling standard INT8 models, operations are typically mapped for x64 CPU vector processing layouts. Loading this exact graph structure onto an NVIDIA GPU via the CUDA execution provider throws a specific warning flag:

```
[W:onnxruntime:] 53 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance.

```

Because the underlying operators do not map natively to the GPU's default execution layout, the engine is forced to inject **53 distinct memory copy operations** across tensor boundary layers. Data must constantly transform between execution memory formats and specialized hardware registers. This memory bus contention introduces latency that outweighs the raw compute savings of 8-bit integer operations.

---

## 7. Evaluation & Empirical Results

The pipeline was benchmarked end-to-end on an **NVIDIA GeForce RTX 2050 Laptop GPU** running the `CUDAExecutionProvider` backend environment.

### 7.1 Quantitative Optimization Metrics

| Model Layout | Target Hardware Backend | Compute Precision | Target Memory Size | Inference Latency | System Throughput |
| --- | --- | --- | --- | --- | --- |
| **Baseline Simplified** | NVIDIA RTX 2050 GPU | FP32 | 13.34 MB | **3.09 ms** | **324.13 FPS** |
| **Quantized Static (QDQ)** | NVIDIA RTX 2050 GPU | INT8 | **3.48 MB** | 4.40 ms | 227.23 FPS |

### 7.2 Efficiency Summary

* **Memory Footprint Reduction:** **73.9% space saved** (shrinking the asset deployment size by **3.83x**, down from 13.34 MB to 3.48 MB).
* **Execution Profile:** The FP32 model achieved maximum throughput on this specific architecture due to native 32-bit floating-point optimization. The INT8 graph successfully achieved compression goals, but encountered a hardware layout bottleneck that increased processing latency.

---

## 8. Known Limitations

* **Cross-Provider Int8 Slowdowns:** Standard ONNX INT8 structures can encounter processing latency overhead when deployed across mixed CUDA execution environments due to target memory layout constraints.
* **Synthetic Calibration Dependency:** The data calibration reader currently uses synthetic normal distributions to profile activation scales. While functional for architectural tests, this can lead to quantization noise and accuracy drops on real-world datasets compared to using actual production frames.

---

## 9. Production Improvements

* **Implement an Explicit TensorRT Execution Pipeline:** To resolve GPU memory copy overhead, add a dedicated compilation pass using `TensorRTExecutionProvider`. This compiles the graph directly into an optimized native `.engine` binary file, leveraging internal FP16/INT8 precision adjustments for optimal Tensor Core execution.
* **Integrate RAGAS-Style Accuracy Evaluations:** Implement automated precision scoring checks (e.g., Top-1/Top-5 accuracy validation) to continuously monitor accuracy drops caused by INT8 compression steps.
* **Integrate Runtime ONNX Shape Inference Preprocessing:** Call `onnxruntime.quantization.shape_inference.quant_pre_process` before executing quantization passes to assist the compiler in safely merging nested node layers.
* **Incorporate Real-World Calibration Datasets:** Wire the calibration pipeline to an active data source (such as ImageNet validation splits) to ensure activation scales map accurately to real-world edge scenarios.