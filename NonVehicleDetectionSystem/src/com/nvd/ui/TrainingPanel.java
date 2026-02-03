package com.nvd.ui;

import javax.swing.*;
import java.awt.*;

/**
 * Training Panel for model retraining with progress visualization
 */
public class TrainingPanel extends JPanel {
    private JButton startButton;
    private JButton stopButton;
    private JProgressBar progressBar;
    private JTextArea logArea;
    private JLabel statusLabel;
    private JLabel accuracyLabel;
    private JLabel epochLabel;

    public TrainingPanel() {
        setLayout(new BorderLayout(10, 10));
        setBackground(ThemeManager.BG_DARK);
        setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));

        // Top: Control buttons
        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        topPanel.setBackground(ThemeManager.BG_DARK);

        startButton = ThemeManager.createStyledButton("Start Training");
        stopButton = ThemeManager.createStyledButton("Stop Training");
        stopButton.setEnabled(false);

        topPanel.add(startButton);
        topPanel.add(stopButton);

        add(topPanel, BorderLayout.NORTH);

        // Center: Progress and metrics
        JPanel centerPanel = new JPanel(new BorderLayout(10, 10));
        centerPanel.setBackground(ThemeManager.BG_DARK);

        // Status section
        JPanel statusPanel = new JPanel(new GridLayout(3, 2, 10, 10));
        statusPanel.setBackground(ThemeManager.BG_DARK);
        statusPanel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(ThemeManager.ACCENT_COLOR, 1),
                "Training Status",
                0, 0,
                ThemeManager.FONT_HEADING,
                ThemeManager.TEXT_PRIMARY
        ));

        statusLabel = ThemeManager.createStyledLabel("Status: Idle", ThemeManager.FONT_REGULAR);
        epochLabel = ThemeManager.createStyledLabel("Epoch: 0/10", ThemeManager.FONT_REGULAR);
        accuracyLabel = ThemeManager.createStyledLabel("Accuracy: 0.00%", ThemeManager.FONT_REGULAR);

        statusPanel.add(new JLabel("Status:"));
        statusPanel.add(statusLabel);
        statusPanel.add(new JLabel("Epoch:"));
        statusPanel.add(epochLabel);
        statusPanel.add(new JLabel("Accuracy:"));
        statusPanel.add(accuracyLabel);

        centerPanel.add(statusPanel, BorderLayout.NORTH);

        // Progress bar
        progressBar = new JProgressBar(0, 100);
        progressBar.setValue(0);
        progressBar.setStringPainted(true);
        progressBar.setForeground(ThemeManager.SUCCESS_COLOR);
        progressBar.setBackground(ThemeManager.BG_DARK);

        centerPanel.add(progressBar, BorderLayout.CENTER);

        // Log area
        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setBackground(ThemeManager.BG_DARK);
        logArea.setForeground(ThemeManager.TEXT_PRIMARY);
        logArea.setFont(ThemeManager.FONT_REGULAR);
        logArea.setText("Training logs will appear here...\n");

        JScrollPane scrollPane = new JScrollPane(logArea);
        scrollPane.setBackground(ThemeManager.BG_DARK);

        centerPanel.add(scrollPane, BorderLayout.SOUTH);

        add(centerPanel, BorderLayout.CENTER);
    }

    public JButton getStartButton() {
        return startButton;
    }

    public JButton getStopButton() {
        return stopButton;
    }

    public void setTrainingStarted(boolean started) {
        startButton.setEnabled(!started);
        stopButton.setEnabled(started);
    }

    public void updateStatus(String status) {
        statusLabel.setText("Status: " + status);
        statusLabel.revalidate();
    }

    public void updateEpoch(int current, int total) {
        epochLabel.setText("Epoch: " + current + "/" + total);
        progressBar.setValue((current * 100) / total);
        progressBar.revalidate();
    }

    public void updateAccuracy(float accuracy) {
        accuracyLabel.setText(String.format("Accuracy: %.2f%%", accuracy * 100));
        accuracyLabel.revalidate();
    }

    public void addLog(String message) {
        logArea.append(message + "\n");
        logArea.setCaretPosition(logArea.getDocument().getLength());
    }

    public void clearLogs() {
        logArea.setText("Training logs will appear here...\n");
    }
}
