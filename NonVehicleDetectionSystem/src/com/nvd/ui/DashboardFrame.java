package com.nvd.ui;

import javax.swing.*;
import java.awt.*;

import com.nvd.utils.Constants;

public class DashboardFrame extends JFrame {

    private CardLayout cardLayout;
    private JPanel mainPanel;

    public DashboardFrame() {

        setTitle(Constants.APP_TITLE);
        setSize(1100, 700);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        ThemeManager.applyTheme();

        cardLayout = new CardLayout();
        mainPanel = new JPanel(cardLayout);

        mainPanel.add(new ImageUploadPanel(), "IMAGE");
        mainPanel.add(new CameraPanel(), "CAMERA");

        add(createTopBar(), BorderLayout.NORTH);
        add(mainPanel, BorderLayout.CENTER);

        setVisible(true);
    }

    private JPanel createTopBar() {

        JPanel bar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        bar.setBackground(new Color(30, 30, 30));

        JButton imageBtn = new JButton("Upload Image");
        JButton cameraBtn = new JButton("Live Camera");

        imageBtn.addActionListener(e -> cardLayout.show(mainPanel, "IMAGE"));
        cameraBtn.addActionListener(e -> cardLayout.show(mainPanel, "CAMERA"));

        bar.add(imageBtn);
        bar.add(cameraBtn);

        return bar;
    }
}
