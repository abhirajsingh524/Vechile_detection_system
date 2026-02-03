package com.nvd.ui;

import javax.swing.*;
import java.awt.*;

public class ThemeManager {

    public static final Color BG_DARK = new Color(25, 25, 25);
    public static final Color ACCENT_COLOR = new Color(0, 153, 255);
    public static final Color TEXT_PRIMARY = Color.WHITE;
    public static final Color SUCCESS_COLOR = new Color(76, 175, 80);
    public static final Font FONT_HEADING = new Font("Arial", Font.BOLD, 14);
    public static final Font FONT_REGULAR = new Font("Arial", Font.PLAIN, 12);

    public static void applyTheme() {
        UIManager.put("Panel.background", BG_DARK);
        UIManager.put("Button.background", new Color(45, 45, 45));
        UIManager.put("Button.foreground", TEXT_PRIMARY);
        UIManager.put("Label.foreground", TEXT_PRIMARY);
        UIManager.put("OptionPane.background", new Color(30, 30, 30));
    }

    public static JButton createStyledButton(String text) {
        JButton btn = new JButton(text);
        btn.setBackground(new Color(45, 45, 45));
        btn.setForeground(TEXT_PRIMARY);
        btn.setFont(FONT_REGULAR);
        return btn;
    }

    public static JLabel createStyledLabel(String text, Font font) {
        JLabel lbl = new JLabel(text);
        lbl.setForeground(TEXT_PRIMARY);
        if (font != null) lbl.setFont(font);
        return lbl;
    }
}
