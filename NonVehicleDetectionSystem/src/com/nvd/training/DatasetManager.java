package com.nvd.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class DatasetManager {

    private static final String IMPROVEMENT_DATASET_PATH =
            "dataset/processed/improvement/";

    /**
     * Save misclassified image for retraining
     */
    public static void saveForImprovement(File imageFile, String correctLabel) {

        try {
            File targetDir = new File(IMPROVEMENT_DATASET_PATH + correctLabel);
            if (!targetDir.exists()) {
                targetDir.mkdirs();
            }

            File targetFile = new File(targetDir, imageFile.getName());
            Files.copy(imageFile.toPath(), targetFile.toPath(),
                    StandardCopyOption.REPLACE_EXISTING);

        } catch (IOException e) {
            System.err.println("Failed to save image for improvement: " + e.getMessage());
        }
    }
}
