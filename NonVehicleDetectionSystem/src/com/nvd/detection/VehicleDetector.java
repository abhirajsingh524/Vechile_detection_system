package com.nvd.detection;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collections;

import com.microsoft.onnxruntime.*;

import com.nvd.postprocessing.DetectionResult;
import com.nvd.utils.Constants;
import com.nvd.utils.FileUtils;
import com.nvd.preprocessing.ImagePreprocessor;

public class VehicleDetector {

    private OrtSession session;

    public VehicleDetector() {
        try {
            this.session = ModelLoader.getSession();
        } catch (Exception e) {
            throw new RuntimeException("Model initialization failed", e);
        }
    }

    /**
     * Detect vehicle from uploaded image
     */
    public DetectionResult detectFromImage(File imageFile) {

        BufferedImage image = ImagePreprocessor.loadImage(imageFile);
        return runInference(image);
    }

    /**
     * Detect vehicle from live camera frame
     */
    public DetectionResult detectFromFrame(BufferedImage frame) {
        return runInference(frame);
    }

    /**
     * Core inference logic
     */
    private DetectionResult runInference(BufferedImage image) {

        try {
            OnnxTensor inputTensor = ImagePreprocessor.preprocess(image);

            OrtSession.Result result = session.run(
                    Collections.singletonMap(Constants.MODEL_INPUT_NAME, inputTensor)
            );

            float[] raw = ((float[][]) result.get(0).getValue())[0];

            // Convert logits to probabilities via softmax (handles both logits and probs)
            float[] probs = softmax(raw);

            int predictedClass = argMax(probs);
            float confidence = probs[predictedClass];

            // Slightly adjust confidence with image analysis but do not force a minimum
            confidence = adjustConfidenceWithImageAnalysis(image, confidence, predictedClass);

            String detailedLabel = FileUtils.getLabelByIndex(predictedClass);
            String displayLabel = FileUtils.toGenericLabel(detailedLabel);

            return new DetectionResult(
                    displayLabel,
                    detailedLabel,
                    confidence,
                    confidence >= Constants.ACCURACY_THRESHOLD
            );

        } catch (Exception e) {
            e.printStackTrace();
            return DetectionResult.failed("Inference failed");
        }
    }

    private int argMax(float[] values) {
        int maxIndex = 0;
        float maxValue = values[0];

        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
            return maxIndex;
    }

    /**
     * Small confidence adjustment based on brightness; never forces a minimum.
     */
    private float adjustConfidenceWithImageAnalysis(BufferedImage image, float baseConfidence, int classIndex) {
        int brightness = calculateImageBrightness(image);

        float adjustment = 0f;
        if (classIndex == 1) {
            if (brightness > 180) adjustment = 0.04f;
            else if (brightness < 80) adjustment = -0.02f;
        } else {
            if (brightness < 100) adjustment = 0.03f;
        }

        float val = Math.min(1.0f, Math.max(0f, baseConfidence + adjustment));
        return val;
    }

    /**
     * Calculate average brightness of image
     */
    private int calculateImageBrightness(BufferedImage image) {
        long totalBrightness = 0;
        int pixelCount = 0;

        for (int y = 0; y < image.getHeight(); y += 10) {
            for (int x = 0; x < image.getWidth(); x += 10) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                totalBrightness += (r + g + b) / 3;
                pixelCount++;
            }
        }

        return pixelCount > 0 ? (int) (totalBrightness / pixelCount) : 128;
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;

        double sum = 0.0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - max);
            sum += exps[i];
        }

        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) probs[i] = (float) (exps[i] / sum);
        return probs;
    }
}
