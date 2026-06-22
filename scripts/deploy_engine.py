import numpy as np
import onnxruntime as ort
import cv2

class EdgeInferenceEngine:
    def __init__(self, model_path="models/onnx/mobilenet_v2_int8.onnx"):
        # Load the optimized engine into memory
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        # Resize raw image to match input layers (224x224)
        resized = cv2.resize(frame, (224, 224))
        # Convert color layout from BGR (OpenCV default) to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values from 0-255 down to a 0-1 decimal range
        normalized = rgb.astype(np.float32) / 255.0
        # Rearrange dimensions from (H, W, C) to channels-first (C, H, W)
        channels_first = np.transpose(normalized, (2, 0, 1))
        # Add a batch dimension to create the expected 4D array shape (1, C, H, W)
        input_tensor = np.expand_dims(channels_first, axis=0)
        return input_tensor

    def predict(self, frame):
        input_tensor = self.preprocess(frame)
        # Execute forward inference pass
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]

if __name__ == "__main__":
    print("\nInitializing Production Inference Engine Wrapper")
    engine = EdgeInferenceEngine()
    
    # Generate a dummy image matrix simulating a live camera frame input
    fake_camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    predictions = engine.predict(fake_camera_frame)
    print(" Success: Inference wrapper executed cleanly.")
    print(f"   Output Prediction Vector Shape: {predictions.shape}")