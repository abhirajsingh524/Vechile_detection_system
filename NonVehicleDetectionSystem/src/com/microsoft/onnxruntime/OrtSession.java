package com.microsoft.onnxruntime;

import java.util.Map;

/**
 * Mock OrtSession for compilation
 */
public class OrtSession {
    
    public static class SessionOptions {
        public enum OptLevel { ALL_OPT }
        
        public void setOptimizationLevel(OptLevel level) {
            // Mock implementation
        }
    }
    
    public static class Result {
        private Object[] values;
        
        public Result(Object[] values) {
            this.values = values;
        }
        
        public ResultWrapper get(int index) {
            return new ResultWrapper(values[index]);
        }
    }
    
    public static class ResultWrapper {
        private Object value;
        
        public ResultWrapper(Object value) {
            this.value = value;
        }
        
        public Object getValue() {
            return value;
        }
    }
    
    public Result run(Map<String, OnnxTensor> inputs) {
        // Mock: return dummy output
        float[][] output = {{0.8f, 0.2f}};
        return new Result(new Object[]{output});
    }
    
    public void close() {
        // Mock implementation
    }
}
