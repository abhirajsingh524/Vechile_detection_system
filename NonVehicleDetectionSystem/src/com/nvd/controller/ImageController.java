package com.nvd.controller;

import java.io.File;

import javax.swing.JOptionPane;

import com.nvd.postprocessing.DetectionResult;

public class ImageController {

    private final DetectionController detectionController;

    public ImageController() {
        this.detectionController = new DetectionController();
    }

    /**
     * Validate and process image
     */
    public DetectionResult processImage(File imageFile) {

        if (imageFile == null || !imageFile.exists()) {
            JOptionPane.showMessageDialog(null, "Invalid image file selected");
            return DetectionResult.failed("Invalid image file");
        }

        return detectionController.detectFromImage(imageFile);
    }
}
