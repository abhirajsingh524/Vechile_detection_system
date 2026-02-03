package com.nvd.preprocessing;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import com.microsoft.onnxruntime.OnnxTensor;
import com.microsoft.onnxruntime.OrtEnvironment;

public class ImagePreprocessor {

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;

    /**
     * Load image from file
     */
    public static BufferedImage loadImage(File file) {
        try {
            return ImageIO.read(file);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load image", e);
        }
    }

    /**
     * Convert BufferedImage → ONNX Tensor
     */
    public static OnnxTensor preprocess(BufferedImage image) {

        try {
            BufferedImage resized = resize(image, IMAGE_WIDTH, IMAGE_HEIGHT);

            float[][][][] input = new float[1][3][IMAGE_HEIGHT][IMAGE_WIDTH];

            for (int y = 0; y < IMAGE_HEIGHT; y++) {
                for (int x = 0; x < IMAGE_WIDTH; x++) {

                    int pixel = resized.getRGB(x, y);

                    float r = ((pixel >> 16) & 0xFF) / 255.0f;
                    float g = ((pixel >> 8) & 0xFF) / 255.0f;
                    float b = (pixel & 0xFF) / 255.0f;

                    input[0][0][y][x] = r;
                    input[0][1][y][x] = g;
                    input[0][2][y][x] = b;
                }
            }

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            return OnnxTensor.createTensor(env, input);

        } catch (Exception e) {
            throw new RuntimeException("Image preprocessing failed", e);
        }
    }

    private static BufferedImage resize(BufferedImage image, int width, int height) {

        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }
}
