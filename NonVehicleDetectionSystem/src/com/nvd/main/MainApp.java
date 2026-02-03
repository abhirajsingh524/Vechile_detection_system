package com.nvd.main;

import java.io.File;

import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import com.nvd.ui.DashboardFrame;
import com.nvd.controller.CameraController;
import com.nvd.detection.VehicleDetector;
import com.nvd.utils.ReportUtil;
import com.nvd.postprocessing.DetectionResult; 

public class MainApp {

    public static void main(String[] args) {

        try {
            // Camera/OpenCV optional: continue even if OpenCV isn't available
            System.out.println("OpenCV initialization skipped or not available; camera will be simulated if needed.");

            // 2️⃣ Verify model exists
            verifyModel();

            // If an image path or image name is provided as an argument, run detection on it and exit
            if (args != null && args.length > 0) {
                String first = args[0];

                if ("--camera".equalsIgnoreCase(first) || "camera".equalsIgnoreCase(first)) {
                    // Run a single-frame camera capture and detection
                    CameraController camCtrl = new CameraController();
                    try {
                        camCtrl.startCamera();
                        com.nvd.postprocessing.DetectionResult result = camCtrl.detectFromCamera();
                        if (result == null || result.hasFailed()) {
                            System.out.println("Camera detection failed: " + (result == null ? "no frame" : result.getMessage()));
                        } else {
                            String out = String.format("Camera capture -> Vehicle: %s | Category: %s | Accuracy Rate: %.2f%% | Accurate: %s",
                                    result.getDetailedLabel(),
                                    result.getLabel(),
                                    result.getAccuracyRate(),
                                    result.isAccurate() ? "YES" : "NO");
                            System.out.println(out);
                            // Save captured image (if available) and report
                            try {
                                CameraController cc = camCtrl;
                                java.awt.image.BufferedImage last = cc.getLastFrame();
                                if (last != null) {
                                    java.io.File imgDir = new java.io.File("output/detected_images");
                                    imgDir.mkdirs();
                                    java.io.File imgOut = new java.io.File(imgDir, "camera_capture.png");
                                    javax.imageio.ImageIO.write(last, "PNG", imgOut);
                                    System.out.println("Saved camera image to: " + imgOut.getAbsolutePath());
                                }
                            } catch (Exception e) {
                                System.err.println("Failed to save camera capture image: " + e.getMessage());
                            }

                            com.nvd.utils.ReportUtil.saveDetectionReport("camera_capture.png", result);
                        }
                    } catch (Exception e) {
                        System.err.println("Error using camera: " + e.getMessage());
                    } finally {
                        camCtrl.stopCamera();
                    }
                    return;
                }

                // Special CLI: export model package into a ZIP
                if ("--export-model".equalsIgnoreCase(first) || "export-model".equalsIgnoreCase(first)) {
                    try {
                        java.io.File out = new java.io.File("output/model_package.zip");
                        com.nvd.utils.PackagingUtil.exportModelPackage(out);
                        System.out.println("Model package exported: " + out.getAbsolutePath());
                    } catch (Exception e) {
                        System.err.println("Failed to export model package: " + e.getMessage());
                    }
                    return;
                }

                String imageArg = first;
                File imageFile = resolveImageFile(imageArg);
                if (imageFile == null) {
                    System.out.println("Image '" + imageArg + "' not found in dataset. Creating a sample image for demo.");
                    imageFile = createSampleImage();
                    System.out.println("Sample image created at: " + imageFile.getAbsolutePath());
                }

                VehicleDetector detector = new VehicleDetector();
                com.nvd.postprocessing.DetectionResult result = detector.detectFromImage(imageFile);

                String out = String.format("File: %s -> Vehicle: %s | Category: %s | Accuracy Rate: %.2f%% | Accurate: %s",
                        imageFile.getName(),
                        result.getDetailedLabel(),
                        result.getLabel(),
                        result.getAccuracyRate(),
                        result.isAccurate() ? "YES" : "NO");

                System.out.println(out);
                com.nvd.utils.ReportUtil.saveDetectionReport(imageFile.getName(), result);
                return; // exit after CLI detection
            }

            // 3️⃣ Launch UI safely
            SwingUtilities.invokeLater(() -> {
                new DashboardFrame();
            });

        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(
                    null,
                    "Application failed to start:\n" + e.getMessage(),
                    "Startup Error",
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    /**
     * Verify trained ONNX model exists
     */
    private static void verifyModel() {

        String[] candidates = new String[] {
                "model/vehicle_detector.onnx",
                "NonVehicleDetectionSystem/model/vehicle_detector.onnx",
                "../model/vehicle_detector.onnx"
        };

        File found = null;
        for (String p : candidates) {
            File f = new File(p);
            System.out.println("Checking model at: " + f.getAbsolutePath() + " exists=" + f.exists());
            if (f.exists()) {
                found = f;
                break;
            }
        }

        if (found == null) {
            throw new RuntimeException("Trained model not found. Checked candidate locations.");
        }

        System.out.println("Model verified: " + found.getAbsolutePath());
    }

    /**
     * Resolve a given image name or path by searching dataset folders
     */
    private static File resolveImageFile(String nameOrPath) {
        // Direct path
        File f = new File(nameOrPath);
        if (f.exists() && f.isFile()) return f;

        // Look into dataset raw folders
        String[] folders = new String[] {"dataset/raw/vehicle", "dataset/raw/non_vehicle", "resources/sample_images"};
        for (String folder : folders) {
            File candidate = new File(folder, nameOrPath);
            if (candidate.exists() && candidate.isFile()) return candidate;
        }

        // Also try without extension
        for (String folder : folders) {
            for (String ext : new String[]{".jpg", ".png", ".jpeg"}) {
                File candidate = new File(folder, nameOrPath + ext);
                if (candidate.exists() && candidate.isFile()) return candidate;
            }
        }
        return null;
    }

    /**
     * Create a simple synthetic image to demo detection when no image is provided
     */
    private static File createSampleImage() {
        try {
            java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(224, 224, java.awt.image.BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g = img.createGraphics();
            g.setColor(java.awt.Color.GRAY);
            g.fillRect(0, 0, 224, 224);
            g.setColor(java.awt.Color.RED);
            g.fillOval(56, 56, 112, 112);
            g.dispose();

            File outDir = new File("output/detected_images");
            outDir.mkdirs();
            File out = new File(outDir, "sample_input.png");
            javax.imageio.ImageIO.write(img, "PNG", out);
            return out;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create sample image", e);
        }
    }


}
