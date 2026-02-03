package com.nvd.preprocessing;

/**
 * Normalization utilities for image preprocessing
 */
public class NormalizationUtils {

    private NormalizationUtils() {
        // Prevent instantiation
    }

    /**
     * Normalize pixel values from [0, 255] to [0, 1]
     */
    public static float normalizePixel(int pixelValue) {
        return pixelValue / 255.0f;
    }

    /**
     * Normalize with ImageNet mean/std (optional)
     */
    public static float normalizeWithImageNet(float pixelValue, float mean, float std) {
        return (pixelValue - mean) / std;
    }

    /**
     * ImageNet normalization constants (BGR format)
     */
    public static class ImageNetConstants {
        public static final float MEAN_B = 103.939f;
        public static final float MEAN_G = 116.779f;
        public static final float MEAN_R = 123.68f;
        
        public static final float STD_B = 1.0f;
        public static final float STD_G = 1.0f;
        public static final float STD_R = 1.0f;
    }

    /**
     * Normalize RGB channels for common models
     */
    public static float[] normalizeRGB(int r, int g, int b) {
        return new float[]{
            normalizePixel(r),
            normalizePixel(g),
            normalizePixel(b)
        };
    }

    /**
     * Min-Max scaling
     */
    public static float minMaxScale(float value, float min, float max) {
        return (value - min) / (max - min);
    }
}
