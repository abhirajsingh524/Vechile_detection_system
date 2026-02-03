package com.nvd.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;

import com.nvd.postprocessing.DetectionResult;

public final class ReportUtil {

    private ReportUtil() {}

    public static void saveDetectionReport(String filename, DetectionResult result) {
        try {
            File reports = new File("output/reports");
            reports.mkdirs();
            File out = new File(reports, "detections.csv");
            boolean exists = out.exists();
            try (FileWriter fw = new FileWriter(out, true)) {
                if (!exists) fw.write("timestamp,filename,label,confidence,accurate\n");
                fw.write(String.format("%s,%s,%s,%.4f,%s\n", Instant.now().toString(), filename, result.getLabel(), result.getConfidence(), result.isAccurate()));
            }
        } catch (IOException e) {
            System.err.println("Warning: failed to save detection report: " + e.getMessage());
        }
    }
}