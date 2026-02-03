package com.nvd.postprocessing;

/**
 * Formats detection results for display and export
 */
public class ResultFormatter {

    private ResultFormatter() {
        // Prevent instantiation
    }

    /**
     * Format detection result as human-readable string
     */
    public static String formatResult(DetectionResult result) {
        if (result == null) {
            return "No detection result";
        }
        
        if (result.hasFailed()) {
            return "Detection Error: " + result.getMessage();
        }
        
        return String.format(
            "Object: %s | Confidence: %.2f%% | Status: %s",
            result.getLabel(),
            result.getConfidence() * 100,
            result.isAccurate() ? "Confident" : "Low Confidence"
        );
    }

    /**
     * Format for logging
     */
    public static String formatForLog(DetectionResult result) {
        if (result == null || result.hasFailed()) {
            return "FAILED";
        }
        return String.format("%s|%.4f|%s", result.getLabel(), result.getConfidence(), result.isAccurate());
    }

    /**
     * Format for CSV export
     */
    public static String formatForCSV(String filename, DetectionResult result) {
        if (result == null || result.hasFailed()) {
            return String.format("\"%s\",FAILED,0,false", filename);
        }
        return String.format("\"%s\",\"%s\",%.4f,%s", 
            filename, 
            result.getLabel(), 
            result.getConfidence(),
            result.isAccurate());
    }
}
