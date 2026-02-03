package com.nvd.training;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;

public class RetrainingHelper {

    private static final String REPORT_PATH =
            "output/reports/retraining_report.txt";

    /**
     * Generate retraining report
     */
    public static void generateReport(String notes) {

        try (FileWriter writer = new FileWriter(REPORT_PATH, true)) {

            writer.write("=== Retraining Report ===\n");
            writer.write("Date: " + LocalDateTime.now() + "\n");
            writer.write(notes + "\n");
            writer.write("-------------------------\n\n");

        } catch (IOException e) {
            System.err.println("Failed to write retraining report");
        }
    }
}
