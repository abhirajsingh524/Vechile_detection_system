package com.nvd.detection;

import java.io.File;
import java.awt.image.BufferedImage;

import com.nvd.postprocessing.DetectionResult;

public class InferenceEngine {

    private VehicleDetector vehicleDetector;

    public InferenceEngine() {
        this.vehicleDetector = new VehicleDetector();
    }

    /**
     * Run inference on uploaded image
     */
    public DetectionResult runImageInference(File imageFile) {

        // TODO: convert image file to tensor (later)
        return vehicleDetector.detectFromImage(imageFile);
    }

    /**
     * Run inference on live camera frame
     */
    public DetectionResult runFrameInference(BufferedImage frame) {

        // TODO: convert frame to tensor (later)
        return vehicleDetector.detectFromFrame(frame);
    }
}
