import os
import time
import numpy as np
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort

def export_mobilenet_to_onnx(output_path="models/onnx/mobilenet_v2.onnx"):
    """
    Loads a pre-trained MobileNetV2 model from PyTorch and exports it 
    to the standardized ONNX format.
    """
    print("\n 1.) Loading Pre-trained MobileNetV2")
    # Load the lightweight model built for edge devices
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()  # Set to evaluation mode (turns off dropout, batchnorm tracking)

    # Create dummy input simulating a single 3-channel (RGB) image of 224x224 pixels
    # Syntax: torch.randn(Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("2.) Exporting to ONNX Format")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,        # Store the trained weights inside the model file
        opset_version=18,          # Use a stable, widely supported hardware operator set #have updated the code to use opset_version18
        do_constant_folding=True,  # Optimize by pre-calculating constant operations
        input_names=['input'],     # Name the input node for the deployment engine
        output_names=['output'],   # Name the output node
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allow flexible batch sizes
    )
    print(f"Model saved locally at: {output_path}")

def profile_onnx_latency(model_path):
    """
    Loads the newly created ONNX model using ONNX Runtime and benchmarks 
    its baseline inference speed on the CPU.
    """
    print("3.) Benchmarking Baseline CPU Latency")
    
    # Initialize the runtime session specifically on the CPU
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Generate random input data matching the required shape
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print("Warming up the processor engine (10 iterations)...")
    # Warm-up runs prevent initial CPU hardware spin-up delays from skewing results
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    print("Running main benchmark loop (100 iterations)...")
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_data})
    end_time = time.time()

    # Calculate average time taken per image in milliseconds
    latency_ms = (end_time - start_time) / iterations * 1000
    print(f"\nBASELINE BENCHMARK RESULTS:")
    print(f"   Average Latency: {latency_ms:.2f} ms per image")
    print(f"   Estimated Throughput: {1000 / latency_ms:.2f} Frames Per Point (FPS)")

if __name__ == "__main__":
    # Ensure the target directory exists before running
    os.makedirs("models/onnx", exist_ok=True)
    
    export_mobilenet_to_onnx()
    profile_onnx_latency("models/onnx/mobilenet_v2.onnx")