package com.nvd.ui;

import javax.swing.*;
import java.awt.*;

import com.nvd.postprocessing.DetectionResult;

public class ResultPanel extends JPanel {

    private JLabel filenameLabel;
    private JLabel vehicleTypeLabel;
    private JLabel genericLabel;
    private JLabel accuracyRateLabel;
    private JLabel status;

    public ResultPanel() {

        setLayout(new GridLayout(3, 2));
        setBackground(new Color(20, 20, 20));

        filenameLabel = createLabel("File: -");
        vehicleTypeLabel = createLabel("Vehicle Type: -");
        genericLabel = createLabel("Category: -");
        accuracyRateLabel = createLabel("Accuracy Rate: -");
        status = createLabel("Status: -");

        add(filenameLabel);
        add(status);
        add(vehicleTypeLabel);
        add(accuracyRateLabel);
        add(genericLabel);
        add(createLabel(""));
    }

    public void updateResult(DetectionResult result) {
        updateResult(result, "-");
    }

    public void updateResult(DetectionResult result, String filename) {

        filenameLabel.setText("File: " + filename);

        if (result.hasFailed()) {
            vehicleTypeLabel.setText("Vehicle Type: Error");
            genericLabel.setText("Category: -");
            accuracyRateLabel.setText("Accuracy Rate: -");
            status.setText(result.getMessage());
            return;
        }

        vehicleTypeLabel.setText("Vehicle Type: " + result.getDetailedLabel());
        genericLabel.setText("Category: " + result.getLabel());
        accuracyRateLabel.setText(String.format("Accuracy Rate: %.2f%%", result.getAccuracyRate()));
        status.setText(result.isAccurate() ? "✓ Accurate" : "⚠ Low Confidence");
    }

    private JLabel createLabel(String text) {
        JLabel lbl = new JLabel(text, JLabel.CENTER);
        lbl.setForeground(Color.WHITE);
        return lbl;
    }
}
