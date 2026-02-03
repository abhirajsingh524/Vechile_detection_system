package com.nvd.training;

import com.nvd.postprocessing.DetectionResult;

public class ImprovementTracker {

    private static int lowConfidenceCount = 0;
    private static int wrongPredictionCount = 0;

    private static final int RETRAIN_THRESHOLD = 20;

    /**
     * Analyze detection result
     */
    public static void track(DetectionResult result) {

        if (result.hasFailed()) return;

        if (!result.isAccurate()) {
            lowConfidenceCount++;
        }
    }

    /**
     * Track wrong prediction explicitly
     */
    public static void trackWrongPrediction() {
        wrongPredictionCount++;
    }

    /**
     * Check if retraining is recommended
     */
    public static boolean shouldRetrain() {
        return (lowConfidenceCount + wrongPredictionCount) >= RETRAIN_THRESHOLD;
    }

    public static String getImprovementStatus() {
        return "Low confidence: " + lowConfidenceCount +
               ", Wrong predictions: " + wrongPredictionCount;
    }

    public static void reset() {
        lowConfidenceCount = 0;
        wrongPredictionCount = 0;
    }
}
