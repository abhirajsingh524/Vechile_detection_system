package com.nvd.utils;

public class AccuracyCalculator {

    private static int totalPredictions = 0;
    private static int correctPredictions = 0;

    /**
     * Update accuracy stats
     */
    public static void update(boolean isCorrect) {
        totalPredictions++;
        if (isCorrect) {
            correctPredictions++;
        }
    }

    /**
     * Get current accuracy
     */
    public static float getAccuracy() {
        if (totalPredictions == 0) return 0f;
        return (float) correctPredictions / totalPredictions;
    }

    /**
     * Reset accuracy (after retraining)
     */
    public static void reset() {
        totalPredictions = 0;
        correctPredictions = 0;
    }
}
