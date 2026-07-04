# scripts/benchmark.py
import os
import sys
import time
import numpy as np

# --- WINDOWS PATH PATCH FOR VENV CUDA ---
# This looks for the deep nvidia directories inside your venv and maps them to Windows
if sys.platform == "win32":
    venv_base = sys.prefix
    nvidia_base = os.path.join(venv_base, "Lib", "site-packages", "nvidia")
    if os.path.exists(nvidia_base):
        for root, dirs, files in os.walk(nvidia_base):
            if "bin" in dirs or "lib" in dirs:
                target_dir = os.path.join(root, "bin" if "bin" in dirs else "lib")
                # Add to both Windows Environment PATH and Python DLL search spaces
                os.environ["PATH"] = target_dir + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(target_dir)
                except AttributeError:
                    pass

# Now we can safely import onnxruntime
import onnxruntime as ort

def benchmark_engine(model_path, name):
    # Prioritize official NVIDIA CUDA execution paths
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    print(f"\nDeploying session for [{name}]...")
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"   Initialization failed with provider layout. Falling back. Error: {e}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    active_backend = session.get_providers()[0]
    print(f"   Active Execution Backend: {active_backend}")
    
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warm-up (10 runs)
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    # Benchmark loop
    iterations = 200
    start_time = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_data})
    end_time = time.time()

    latency = (end_time - start_time) / iterations * 1000
    throughput = 1000 / latency
    file_size = os.path.getsize(model_path) / (1024 * 1024)

    print(f"   Model Size : {file_size:.2f} MB")
    print(f"   Latency    : {latency:.2f} ms per image")
    print(f"   Throughput : {throughput:.2f} FPS")
    
    return latency, throughput, file_size

if __name__ == "__main__":
    fp32_path = "models/onnx/mobilenet_v2_simplified.onnx"
    int8_path = "models/onnx/mobilenet_v2_int8.onnx"

    print("\n================ FINAL EVALUATION ENGINE ================")
    fp32_lat, fp32_fps, fp32_size = benchmark_engine(fp32_path, "Baseline FP32 Model")
    int8_lat, int8_fps, int8_size = benchmark_engine(int8_path, "Quantized INT8 Model")
    
    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(f" Performance Delta : {fp32_lat / int8_lat:.2f}x speedup relative factor")
    print(f" Memory Reduction  : {((fp32_size - int8_size) / fp32_size) * 100:.1f}% space saved")