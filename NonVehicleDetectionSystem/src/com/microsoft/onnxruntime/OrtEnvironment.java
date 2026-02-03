package com.microsoft.onnxruntime;

/**
 * Mock OrtEnvironment for compilation
 */
public class OrtEnvironment {
    
    private static OrtEnvironment instance;
    
    private OrtEnvironment() {
        // Prevent instantiation
    }
    
    public static OrtEnvironment getEnvironment() {
        if (instance == null) {
            instance = new OrtEnvironment();
        }
        return instance;
    }
    
    public OrtSession createSession(String modelPath, OrtSession.SessionOptions options) throws Exception {
        // Mock: return a dummy session
        return new OrtSession();
    }
    
    public void close() {
        // Mock implementation
    }
}
