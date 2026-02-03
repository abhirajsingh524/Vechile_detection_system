package com.nvd.detection;

import java.io.File;
import java.nio.file.Paths;

import com.microsoft.onnxruntime.*;

import com.nvd.exception.ModelLoadException;

public class ModelLoader {

    private static OrtEnvironment environment;
    private static OrtSession session;

    private static final String MODEL_PATH =
            Paths.get("model", "vehicle_detector.onnx").toString();

    private ModelLoader() {
        // Prevent instantiation
    }

    /**
     * Load ONNX model only once
     */
    public static OrtSession getSession() throws ModelLoadException {

        if (session == null) {
            try {
                environment = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

                File modelFile = new File(MODEL_PATH);

                if (!modelFile.exists()) {
                    // try alternative candidate locations
                    String[] candidates = new String[] {
                            "model/vehicle_detector.onnx",
                            "NonVehicleDetectionSystem/model/vehicle_detector.onnx",
                            "../model/vehicle_detector.onnx"
                    };
                    boolean found = false;
                    for (String p : candidates) {
                        File alt = new File(p);
                        if (alt.exists()) {
                            modelFile = alt;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        throw new ModelLoadException("Model file not found at: " + MODEL_PATH);
                    }
                }

                session = environment.createSession(modelFile.getAbsolutePath(), options);

                System.out.println("ONNX model loaded successfully: " + modelFile.getAbsolutePath());

            } catch (Exception e) {
                throw new ModelLoadException("Failed to load ONNX model.", e);
            }
        }
        return session;
    }

    /**
     * Optional cleanup
     */
    public static void close() {
        try {
            if (session != null) session.close();
            if (environment != null) environment.close();
        } catch (Exception ignored) {
        }
    }
}
