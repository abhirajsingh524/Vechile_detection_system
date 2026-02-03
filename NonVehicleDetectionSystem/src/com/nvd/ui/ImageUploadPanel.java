package com.nvd.ui;

import javax.swing.*;
import java.awt.*;
import java.io.File;

import com.nvd.detection.InferenceEngine;
import com.nvd.postprocessing.DetectionResult;

public class ImageUploadPanel extends JPanel {

    private JLabel imageLabel;
    private ResultPanel resultPanel;
    private InferenceEngine engine;

    public ImageUploadPanel() {

        engine = new InferenceEngine();
        setLayout(new BorderLayout());

        imageLabel = new JLabel("No Image Selected", JLabel.CENTER);
        imageLabel.setPreferredSize(new Dimension(400, 400));

        JButton uploadBtn = new JButton("Upload Image & Detect");

        uploadBtn.addActionListener(e -> uploadImage());

        resultPanel = new ResultPanel();

        add(imageLabel, BorderLayout.CENTER);
        add(uploadBtn, BorderLayout.NORTH);
        add(resultPanel, BorderLayout.SOUTH);
    }

    private void uploadImage() {

        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {

            File file = chooser.getSelectedFile();
            imageLabel.setIcon(new ImageIcon(file.getAbsolutePath()));

            DetectionResult result = engine.runImageInference(file);
            resultPanel.updateResult(result, file.getName());
            // Save report entry for uploaded image
            com.nvd.utils.ReportUtil.saveDetectionReport(file.getName(), result);
        }
    }
}
