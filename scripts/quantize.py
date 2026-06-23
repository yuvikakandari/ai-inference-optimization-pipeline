# quantize.py
import os
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class CameraCalibrationDataReader(CalibrationDataReader):
    """Generates synthetic calibration frames to profile internal activation scales"""
    def __init__(self, input_name, shape=(1, 3, 224, 224), samples=10):
        self.data = [{input_name: np.random.randn(*shape).astype(np.float32)} for _ in range(samples)]
        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

def quantize_model_static(input_path="models/onnx/mobilenet_v2_simplified.onnx", output_path="models/onnx/mobilenet_v2_int8.onnx"):
    print("\n--- Starting Post-Training Static INT8 Quantization (PTQ) ---")
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Please run optimize.py first.")
        return

    dr = CameraCalibrationDataReader(input_name="input")

    # Static quantization optimizes both weights AND activations for hardware pipelines
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=QuantType.QInt8,   # Signed Int8 offers cleaner acceleration matrices on modern GPUs
        weight_type=QuantType.QInt8
    )
    
    print(f" Success: Quantized model saved at: {output_path}")
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Size Comparison: {orig_size:.2f} MB down to {quant_size:.2f} MB ({(orig_size/quant_size):.2f}x smaller)")

if __name__ == "__main__":
    quantize_model_static()