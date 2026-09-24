#!/bin/bash
# Auto-generate a unique session name
SESSION="cuda-job-$RANDOM"

echo "Provisioning T4 GPU..."
colab new -s $SESSION --gpu T4

echo "Uploading input and source files..."
colab upload -s $SESSION outputs/gpu_inputs/input.bin /content/input.bin
colab upload -s $SESSION src/cuda/Kernels.cu /content/Kernels.cu

echo "Compiling and Executing..."
# Execute the compilation and run via Python's subprocess over colab exec
echo "
import subprocess
subprocess.run(['nvcc', '-o', '/content/run_kernel', '/content/Kernels.cu'], check=True)
subprocess.run(['/content/run_kernel', '/content/input.bin', '/content/output.bin'], check=True)
print('Execution finished successfully.')
" | colab exec -s $SESSION

echo "Downloading output binary..."
colab download -s $SESSION /content/output.bin outputs/gpu_outputs/output.bin

echo "Cleaning up session..."
colab stop -s $SESSION
echo "All done!"