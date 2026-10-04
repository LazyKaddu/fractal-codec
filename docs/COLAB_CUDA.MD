# Running Fractal Codec CUDA Kernels on Google Colab

This guide explains how to use Google Colab to test and run the CUDA kernels for the Fractal Codec project using the provided Jupyter Notebook (`third_party/fractalCodec_v2.ipynb`). Google Colab provides free access to GPUs, making it an excellent environment for testing CUDA code without needing local GPU hardware.

## Prerequisites
- A Google account.
- The `third_party/fractalCodec_v2.ipynb` notebook from this repository.

## Step 1: Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. You will be greeted with an "Open notebook" window.

## Step 2: Open the Notebook
You can open the notebook directly from GitHub or upload it manually from your local clone.

### Option A: Open directly from GitHub (if the repository is public)
1. In the Colab welcome window (or by clicking **File** > **Open notebook**), select the **GitHub** tab.
2. Enter the URL of this repository.
3. Locate and click on `third_party/fractalCodec_v2.ipynb` to open it.

### Option B: Upload Manually
1. In Colab, go to **File** > **Upload notebook**.
2. Click **Browse** and select your local copy of `third_party/fractalCodec_v2.ipynb` (located at `third_party/fractalCodec_v2.ipynb` in this repo).

## Step 3: Enable GPU Acceleration
By default, Colab might assign a CPU-only runtime. You must enable a GPU to compile and execute the CUDA kernels.
1. In the top menu, click on **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any other available GPU).
3. Click **Save**.

*(Optional)* You can verify that the GPU is allocated by adding a new code cell and running:
```bash
!nvidia-smi
```

## Step 4: Run the Notebook
The notebook contains the necessary commands and code blocks to compile and run the CUDA kernels.
1. Read through the cells to understand what each section does.
2. Run cells sequentially by clicking the **Play** button on the left side of each cell, or by pressing `Shift + Enter`.
3. The notebook handles writing the CUDA C++ code to a file (using the `%%writefile` magic command), compiling it using the `nvcc` compiler, and executing the compiled binary.

## Troubleshooting
- **`nvcc: command not found`**: Make sure you have enabled a GPU runtime in Step 3. The `nvcc` compiler is available by default in GPU environments on Colab.
- **Session Disconnected**: Colab sessions disconnect after a period of inactivity or when resource limits are reached. If this happens, you will need to reconnect, ensure the GPU is still enabled, and re-run your cells from the beginning.
