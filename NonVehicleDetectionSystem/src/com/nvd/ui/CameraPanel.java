package com.nvd.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

import com.nvd.camera.CameraService;
import com.nvd.camera.FrameGrabber;
import com.nvd.detection.InferenceEngine;
import com.nvd.postprocessing.DetectionResult;

public class CameraPanel extends JPanel {

    private JLabel cameraView;
    private ResultPanel resultPanel;
    private CameraService cameraService;
    private InferenceEngine engine;
    private Timer timer;

    public CameraPanel() {

        engine = new InferenceEngine();
        cameraService = new CameraService();

        setLayout(new BorderLayout());

        cameraView = new JLabel();
        resultPanel = new ResultPanel();

        JButton startBtn = new JButton("Start Camera");
        JButton stopBtn = new JButton("Stop Camera");

        startBtn.addActionListener(e -> startCamera());
        stopBtn.addActionListener(e -> stopCamera());

        JPanel controls = new JPanel();
        controls.add(startBtn);
        controls.add(stopBtn);

        add(cameraView, BorderLayout.CENTER);
        add(controls, BorderLayout.NORTH);
        add(resultPanel, BorderLayout.SOUTH);
    }

    private void startCamera() {
        try {
            cameraService.startCamera();

            timer = new Timer(100, e -> {
                var mat = cameraService.getFrame();
                if (mat != null) {
                    BufferedImage image = FrameGrabber.matToBufferedImage(mat);
                    cameraView.setIcon(new ImageIcon(image));

                    DetectionResult result = engine.runFrameInference(image);
                    resultPanel.updateResult(result, "camera_capture.png");
                }
            });
            timer.start();

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, ex.getMessage());
        }
    }

    private void stopCamera() {
        if (timer != null) timer.stop();
        cameraService.stopCamera();
    }
}
