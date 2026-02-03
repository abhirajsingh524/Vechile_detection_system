package com.nvd.controller;

import java.awt.image.BufferedImage;
import java.io.File;

import com.nvd.detection.InferenceEngine;
import com.nvd.postprocessing.DetectionResult;
import com.nvd.utils.AccuracyCalculator;

public class DetectionController {

    private final InferenceEngine inferenceEngine;

    public DetectionController() {
        this.inferenceEngine = new InferenceEngine();
    }

    /**
     * Detect from uploaded image
     */
    public DetectionResult detectFromImage(File imageFile) {

        DetectionResult result = inferenceEngine.runImageInference(imageFile);

        if (!result.hasFailed()) {
            AccuracyCalculator.update(result.isAccurate());
        }

        return result;
    }

    /**
     * Detect from live camera frame
     */
    public DetectionResult detectFromFrame(BufferedImage frame) {

        DetectionResult result = inferenceEngine.runFrameInference(frame);

        if (!result.hasFailed()) {
            AccuracyCalculator.update(result.isAccurate());
        }

        return result;
    }
}
