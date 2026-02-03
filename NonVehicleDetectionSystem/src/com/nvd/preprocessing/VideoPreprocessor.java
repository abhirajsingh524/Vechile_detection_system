package com.nvd.preprocessing;

import java.awt.image.BufferedImage;

/**
 * Video/Frame preprocessing utilities
 */
public class VideoPreprocessor {

    private VideoPreprocessor() {
        // Prevent instantiation
    }

    /**
     * Convert BufferedImage to grayscale
     */
    public static BufferedImage toGrayscale(BufferedImage image) {
        BufferedImage grayscale = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_BYTE_GRAY
        );
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);
                int grayRGB = (gray << 16) | (gray << 8) | gray;
                
                grayscale.setRGB(x, y, grayRGB);
            }
        }
        
        return grayscale;
    }

    /**
     * Detect edges using Sobel operator
     */
    public static BufferedImage detectEdges(BufferedImage image) {
        BufferedImage edges = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_INT_RGB
        );
        
        BufferedImage grayscale = toGrayscale(image);
        int width = grayscale.getWidth();
        int height = grayscale.getHeight();
        
        // Simplified edge detection
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int gx = -grayscale.getRGB(x - 1, y - 1) + grayscale.getRGB(x + 1, y - 1)
                       - 2 * grayscale.getRGB(x - 1, y) + 2 * grayscale.getRGB(x + 1, y)
                       - grayscale.getRGB(x - 1, y + 1) + grayscale.getRGB(x + 1, y + 1);
                
                int gy = grayscale.getRGB(x - 1, y - 1) + 2 * grayscale.getRGB(x, y - 1) + grayscale.getRGB(x + 1, y - 1)
                       - grayscale.getRGB(x - 1, y + 1) - 2 * grayscale.getRGB(x, y + 1) - grayscale.getRGB(x + 1, y + 1);
                
                int magnitude = (int) Math.min(255, Math.sqrt(gx * gx + gy * gy) / 8);
                int edgeRGB = (magnitude << 16) | (magnitude << 8) | magnitude;
                
                edges.setRGB(x, y, edgeRGB);
            }
        }
        
        return edges;
    }

    /**
     * Apply brightness adjustment
     */
    public static BufferedImage adjustBrightness(BufferedImage image, float factor) {
        BufferedImage adjusted = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_INT_RGB
        );
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int r = Math.min(255, (int) (((rgb >> 16) & 0xFF) * factor));
                int g = Math.min(255, (int) (((rgb >> 8) & 0xFF) * factor));
                int b = Math.min(255, (int) ((rgb & 0xFF) * factor));
                
                int adjustedRGB = (r << 16) | (g << 8) | b;
                adjusted.setRGB(x, y, adjustedRGB);
            }
        }
        
        return adjusted;
    }

    /**
     * Flip image horizontally
     */
    public static BufferedImage flipHorizontal(BufferedImage image) {
        BufferedImage flipped = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_INT_RGB
        );
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                flipped.setRGB(image.getWidth() - 1 - x, y, image.getRGB(x, y));
            }
        }
        
        return flipped;
    }
}
