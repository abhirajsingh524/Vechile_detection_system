package com.nvd.postprocessing;

public class DetectionResult {

    private String label;
    private String detailedLabel;  // e.g., "car", "bike", "truck" before generic mapping
    private float confidence;
    private float accuracyRate;    // alias for accuracy display
    private boolean accurate;
    private boolean failed;
    private String message;

    /**
     * Success constructor
     */
    public DetectionResult(String label, float confidence, boolean accurate) {
        this.label = label;
        this.confidence = confidence;
        this.accuracyRate = confidence * 100;
        this.accurate = accurate;
        this.failed = false;
    }

    /**
     * Constructor with detailed label
     */
    public DetectionResult(String label, String detailedLabel, float confidence, boolean accurate) {
        this.label = label;
        this.detailedLabel = detailedLabel;
        this.confidence = confidence;
        this.accuracyRate = confidence * 100;
        this.accurate = accurate;
        this.failed = false;
    }

    /**
     * Failure constructor
     */
    private DetectionResult(String message) {
        this.failed = true;
        this.message = message;
    }

    public static DetectionResult failed(String message) {
        return new DetectionResult(message);
    }

    public String getLabel() {
        return label;
    }

    public String getDetailedLabel() {
        return detailedLabel != null ? detailedLabel : label;
    }

    public float getConfidence() {
        return confidence;
    }

    public float getAccuracyRate() {
        return accuracyRate;
    }

    public boolean isAccurate() {
        return accurate;
    }

    public boolean hasFailed() {
        return failed;
    }

    public String getMessage() {
        return message;
    }

    @Override
    public String toString() {
        if (failed) {
            return "Detection Failed: " + message;
        }
        return String.format("Result: %s (Accuracy Rate: %.2f%%)", label, accuracyRate);
    }
}
