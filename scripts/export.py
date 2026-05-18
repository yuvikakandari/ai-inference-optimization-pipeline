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
    print("\n--- [Step 1] Loading Pre-trained MobileNetV2 ---")
    # Load the lightweight model built for edge devices
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()  # Set to evaluation mode (turns off dropout, batchnorm tracking)

    # Create dummy input simulating a single 3-channel (RGB) image of 224x224 pixels
    # Syntax: torch.randn(Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)