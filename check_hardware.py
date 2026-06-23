# check_hardware.py
import torch
import onnxruntime as ort

print("=== 1. PyTorch Hardware Check ===")
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available for PyTorch: {cuda_available}")

if cuda_available:
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")

print("\n=== 2. ONNX Runtime Execution Providers ===")
# This outputs what ONNX Runtime is capable of interacting with in your current environment
available_providers = ort.get_available_providers()
print(f"Registered Execution Providers in ORT: {available_providers}")

print("\n=== 3. Summary Guidance ===")
if 'CUDAExecutionProvider' in available_providers:
    print("[STATUS] Excellent! Your GPU is fully ready for NVIDIA CUDA acceleration via ONNX.")
elif 'TensorRTExecutionProvider' in available_providers:
    print("[STATUS] Excellent! TensorRT is available on your environment.")
elif 'DmlExecutionProvider' in available_providers:
    print("[STATUS] AMD/Intel/NVIDIA Windows GPU detected via DirectML execution provider.")
else:
    print("[WARNING] Only CPU provider is available to ONNX Runtime.")
    print("👉 ACTION REQUIRED: Run: 'pip uninstall onnxruntime' then 'pip install onnxruntime-gpu'")