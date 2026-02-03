package com.nvd.controller;

/**
 * Non-Vehicle Classifier for determining if detected object is a vehicle or not
 */
public class NonVehicleClassifier {

    private static final float VEHICLE_CONFIDENCE_THRESHOLD = 0.75f;

    /**
     * Classify detection result
     */
    public static boolean isNonVehicle(com.nvd.postprocessing.DetectionResult result) {
        if (result == null || result.hasFailed()) {
            return false;
        }

        String label = result.getLabel();
        float confidence = result.getConfidence();

        // If classified as "Non-Vehicle" with high confidence
        if ("Non-Vehicle".equalsIgnoreCase(label) && confidence >= VEHICLE_CONFIDENCE_THRESHOLD) {
            return true;
        }

        // If classified as "Vehicle" with low confidence, might be non-vehicle
        if ("Vehicle".equalsIgnoreCase(label) && confidence < (1 - VEHICLE_CONFIDENCE_THRESHOLD)) {
            return true;
        }

        return false;
    }

    /**
     * Get classification confidence
     */
    public static float getClassificationConfidence(com.nvd.postprocessing.DetectionResult result) {
        if (result == null || result.hasFailed()) {
            return 0f;
        }
        return result.getConfidence();
    }

    /**
     * Get classification label
     */
    public static String getClassificationLabel(com.nvd.postprocessing.DetectionResult result) {
        if (result == null || result.hasFailed()) {
            return "Unknown";
        }
        return result.getLabel();
    }
}
