package com.microsoft.onnxruntime;

/**
 * Mock ONNX Tensor for compilation without full ONNX runtime
 */
public class OnnxTensor {
    
    private Object data;
    
    public OnnxTensor(Object data) {
        this.data = data;
    }
    
    public static OnnxTensor createTensor(OrtEnvironment env, float[][][][] data) {
        return new OnnxTensor(data);
    }
    
    public Object getData() {
        return data;
    }
}
