#!/bin/bash
# Auto-queue runner: waits for a PID to finish, then runs next experiment
# Usage: bash run_queue.sh <wait_pid> <gpu_id> <script> [args...]
WAIT_PID=$1
GPU_ID=$2
shift 2

echo "[queue] Waiting for PID $WAIT_PID to finish before launching on GPU$GPU_ID..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 10
done
echo "[queue] PID $WAIT_PID done. Launching: $@"
CUDA_VISIBLE_DEVICES=$GPU_ID /opt/miniconda/envs/rtdetr_env/bin/python "$@"
