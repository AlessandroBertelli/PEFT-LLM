# Scalable Local RAG with Llama 3.2 & Unsloth

## 1. Overview
This project implements a scalable, end-to-end pipeline for training and deploying a **General Purpose** Large Language Model (LLM) optimized for consumer-grade CPUs (e.g., Apple M1/M2 or Intel/AMD).

The solution integrates **Unsloth** for memory-efficient fine-tuning, **llama.cpp** for GGUF quantization, and **Streamlit** for a user-friendly RAG (Retrieval-Augmented Generation) interface.

---

## 2. Methodology: Dual-Approach Optimization
To achieve a scalable system, we implemented improvements via two mutually exclusive approaches: **Model-Centric** (Architecture/Hyperparameters) and **Data-Centric** (Dataset Quality).

### (a) Model-Centric Approach: Architecture & Quantization
We prioritized inference latency and throughput on CPU hardware over raw parameter count.

* **Architecture Selection (3B vs 8B):**
    * **Strategy:** We conducted a comparative analysis between **Llama-3.2-3B-Instruct** and **Llama-3.1-8B**.
    * **Observation:** The 8B model requires significantly higher VRAM and computation, resulting in sluggish CPU inference.
    * **Decision:** We selected the **3B model**. Despite being ~60% smaller, it retains strong reasoning capabilities while offering a **2.5x - 4x speedup** in token generation on local hardware.
* **Hyperparameter Tuning:**
    * We utilized **Unsloth** on a standard Google Colab instance (Tesla T4).
    * **Optimization:** We used `r=16` and `lora_alpha=16` with 4-bit quantization (QLoRA) to fit the training process within the 16GB VRAM limit of the T4, ensuring stable convergence without Out-Of-Memory (OOM) errors.
* **Quantization:**
    * Post-training, we converted the model to **GGUF format (Q4_K_M)**. This reduces the memory footprint to ~2.2GB, allowing the LLM to run alongside the RAG vector store in system RAM.

### (b) Data-Centric Approach: Dataset Quality
We replaced standard, noisy datasets with a high-quality, curated alternative to improve instruction adherence.

* **Dataset:** **[FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)**.
* **Improvement:** Unlike basic datasets (e.g., raw Alpaca or ShareGPT) which often contain duplicated or short/incoherent responses, FineTome-100k is filtered for educational value and reasoning quality.
* **Result:** The model produces more coherent, context-aware answers suitable for a general-purpose assistant, surpassing the baseline performance of models trained on raw web-scraped data.

---

## 3. Performance & Experiments

We performed a comparative analysis to justify the selection of the 3B model for local CPU deployment.

| Metric | **Llama 3.2 3B (Selected)** | Llama 3.1 8B (Discarded) | Conclusion |
| :--- | :--- | :--- | :--- |
| **Parameters** | 3.21 Billion | 8.03 Billion | 3B is ~60% smaller |
| **Training Time (1 epoch)** | **~22 hours (T4 GPU)** | ~50+ hours (Est. T4) | 3B is significantly faster to fine-tune |
| **Inference Speed (CPU)** | **High (~15-20 tok/s)** | Low (~4-5 tok/s) | 8B is too slow for interactive chat |
| **RAM Usage (Quantized)** | ~2.2 GB | ~5.8 GB | 3B allows distinct room for RAG processes |
| **Reasoning Quality** | Excellent for general tasks | Marginally better | Diminishing returns on 8B for this use case |

**Verdict:** The **Llama 3.2 3B** is the optimal choice for this specific pipeline, offering the best trade-off between *response quality* and *user experience (latency)*.

---

## 4. Pipeline Architecture

1.  **Training (Cloud):**
    * **Framework:** Unsloth (Optimized QLoRA).
    * **Hardware:** Google Colab Free Tier (Tesla T4).
    * **Output:** LoRA Adapters merged into FP16 base model.

2.  **Conversion (Local):**
    * **Tool:** `llama.cpp`.
    * **Pipeline:** `FP16 Safetensors` $\rightarrow$ `GGUF FP16` $\rightarrow$ `GGUF Q4_K_M`.

3.  **Inference (Local UI):**
    * **Interface:** Streamlit.
    * **Engine:** `llama-cpp-python`.
    * **Memory Management:** Implemented aggressive Garbage Collection (`gc.collect`) to manage RAM between RAG and Chat modes.

---

## 5. Installation & Usage

### Prerequisites
* Python 3.10+
* `git`, `cmake`, `build-essential` (for compiling llama.cpp)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: On MacOS with Apple Silicon, ensure `llama-cpp-python` is built with Metal support)*:
    ```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```

3.  **Download the Model:**
    The application will automatically download the GGUF model from the Hugging Face Hub (`abertekth/model`) on the first run.

### Running the App
```bash
streamlit run main.py