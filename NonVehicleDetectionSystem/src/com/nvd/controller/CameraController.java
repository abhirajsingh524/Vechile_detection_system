package com.nvd.controller;

import java.awt.image.BufferedImage;

import com.nvd.camera.CameraService;
import com.nvd.camera.FrameGrabber;
import com.nvd.exception.CameraAccessException;
import com.nvd.postprocessing.DetectionResult;

public class CameraController {

    private final CameraService cameraService;
    private final DetectionController detectionController;

    public CameraController() {
        this.cameraService = new CameraService();
        this.detectionController = new DetectionController();
    }

    /**
     * Start camera safely
     */
    public void startCamera() throws CameraAccessException {
        cameraService.startCamera();
    }

    /**
     * Stop camera
     */
    public void stopCamera() {
        cameraService.stopCamera();
    }

    /**
     * Get detection result from live frame
     */
    public DetectionResult detectFromCamera() {

        BufferedImage frame = cameraService.getFrame();
        if (frame == null) return DetectionResult.failed("No frame captured");

        BufferedImage image = FrameGrabber.matToBufferedImage(frame);
        return detectionController.detectFromFrame(image);
    }

    public BufferedImage getLastFrame() {
        return cameraService.getFrame();
    }

    public boolean isCameraActive() {
        return cameraService.isActive();
    }
}
