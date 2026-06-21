import os
import onnx
from onnxsim import simplify

def simplify_onnx_model(input_path="models/onnx/mobilenet_v2.onnx", output_path="models/onnx/mobilenet_v2_simplified.onnx"):
    print("\n Loading Raw ONNX Model ---")
    # 1. Load the original ONNX model we created in Phase A
    model = onnx.load(input_path)
    
    print("Optimizing and Simplifying Graph Structure ---")
    # 2. Run the ONNX Simplifier engine on the model graph
    # simplify() returns two things: the new model object, and a true/false success flag
    model_simplified, check = simplify(model)
    
    # 3. Double-check that the simplified graph is still mathematically valid
    assert check, " Error: Simplified ONNX model validation failed!"
    
    # 4. Save the newly polished model file
    onnx.save(model_simplified, output_path)
    print(f" Success: Optimized graph saved locally at: {output_path}")

    # 5. Print a quick structural comparison for your interview logs
    print("\n STRUCTURAL COMPARISON:")
    print(f"   Original Model File Size   : {os.path.getsize(input_path) / (1024*1024):.2f} MB")
    print(f"   Simplified Model File Size : {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    simplify_onnx_model()