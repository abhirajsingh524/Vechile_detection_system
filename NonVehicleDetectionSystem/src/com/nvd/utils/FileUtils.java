package com.nvd.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class FileUtils {

    private static List<String> labels;

    /**
     * Load labels once
     */
    private static void loadLabels() {
        labels = new ArrayList<>();

        // 1) Prefer model labels if present (model may be multiclass)
        if (tryLoadLabelsFromPath("model/labels.txt")) return;

        // 2) Fallback to dataset labels file
        if (tryLoadLabelsFromPath(Constants.LABELS_FILE_PATH)) return;

        // 3) Try model_config.json containing labels array
        try {
            java.io.File cfg = new java.io.File("model/model_config.json");
            if (cfg.exists()) {
                String content = new String(java.nio.file.Files.readAllBytes(cfg.toPath()));
                if (content.contains("\"labels\"")) {
                    // crude JSON parse: look for [ ... ] after "labels"
                    int idx = content.indexOf("\"labels\"");
                    int start = content.indexOf('[', idx);
                    int end = content.indexOf(']', start);
                    if (start > 0 && end > start) {
                        String arr = content.substring(start + 1, end);
                        String[] parts = arr.split(",");
                        for (String p : parts) {
                            p = p.trim();
                            p = p.replaceAll("\"", "").replaceAll("'", "").trim();
                            if (!p.isEmpty()) labels.add(p);
                        }
                        if (!labels.isEmpty()) return;
                    }
                }
            }
        } catch (Exception ignored) {}

        // 4) Fallback: use defaults
        System.err.println("Warning: labels file not found. Using default labels (non_vehicle, vehicle).");
        labels.add("non_vehicle");
        labels.add("vehicle");
    }

    private static boolean tryLoadLabelsFromPath(String path) {
        String[] candidates = new String[] {path, "NonVehicleDetectionSystem/" + path, "../" + path};
        for (String p : candidates) {
            try {
                System.out.println("Trying labels path: " + p);
                java.io.File f = new java.io.File(p);
                System.out.println("Exists: " + f.exists() + " -> " + f.getAbsolutePath());
                if (!f.exists()) continue;
                try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                    String line;
                    while ((line = br.readLine()) != null) labels.add(line.trim());
                }
                if (!labels.isEmpty()) {
                    System.out.println("Loaded labels from: " + f.getAbsolutePath());
                    return true;
                }
            } catch (Exception e) {
                System.out.println("Error trying to load labels from " + p + ": " + e.getMessage());
            }
        }
        return false;
    }

    /**
     * Get label name from index
     */
    public static String getLabelByIndex(int index) {

        if (labels == null) {
            loadLabels();
            System.out.println("Loaded labels: " + labels);
        }

        if (index < 0 || index >= labels.size()) {
            return "Unknown";
        }

        return labels.get(index);
    }

    /**
     * Whether the model uses multiple vehicle classes (and a non_vehicle class)
     */
    public static boolean isMultiClassVehicleModel() {
        if (labels == null) loadLabels();
        return labels.size() > 2 && labels.contains("non_vehicle");
    }

    /**
     * Convert a detailed model label into a generic one: 'vehicle' or 'non_vehicle'.
     */
    public static String toGenericLabel(String label) {
        if (label == null) return "Unknown";
        if (isMultiClassVehicleModel()) {
            if ("non_vehicle".equalsIgnoreCase(label)) return "non_vehicle";
            return "vehicle";
        }
        return label;
    }
}
