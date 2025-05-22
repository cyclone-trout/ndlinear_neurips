# 🚀 Reproducing OPT Pretraining Results

This guide provides step-by-step instructions to set up the environment and reproduce OPT pretraining results on a GPU-enabled server. Ensure you're working within the project root directory.

---

## 🛠️ Environment Setup

1. **Navigate to the project directory**

    ```bash
    cd OPT-pretrain
    ```

2. **Create and activate a new Conda environment**

    ```bash
    conda create -n opt-pretrain python=3.9
    conda activate opt-pretrain
    ```

3. **Install required Python packages**

    ```bash
    # Install dependencies
    python -m pip install -r requirements.txt
    ```

4. **Install CUDA-compatible PyTorch**

    > ⚠️ Make sure to match the versions with your CUDA installation. The following is an example for CUDA 12.4:

    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    ```

5. **Replace the default OPT model implementation**

    > 🛠️ This step is necessary to ensure compatibility with the custom training scripts.

    ```bash
    cp src/modeling_opt.py <your_conda_path>/anaconda3/envs/<your_env_name>/lib/python3.9/site-packages/transformers/models/opt/modeling_opt.py
    ```

---

## ✅ Reproducing Pretraining Results

1. **Navigate to the scripts directory**

    ```bash
    cd scripts
    ```

2. **Run the pretraining script**

- **Single-GPU (Not Recommended):**

  ```bash
  bash pretrain_opt.sh
  ```

- **Multi-GPU (Recommended for large datasets):**

  ```bash
  bash pretrain_opt_ddp.sh
  ```

---