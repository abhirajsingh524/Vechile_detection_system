package com.nvd.postprocessing;

public class BoundingBox {

    private int x;
    private int y;
    private int width;
    private int height;
    private float confidence;
    private String label;

    public BoundingBox(int x, int y, int width, int height, float confidence, String label) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.confidence = confidence;
        this.label = label;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public float getConfidence() {
        return confidence;
    }

    public String getLabel() {
        return label;
    }

    public String toDisplayString() {
        return label + " (" + String.format("%.2f", confidence * 100) + "%)";
    }
}
