package com.nvd.utils;

public final class Constants {

    private Constants() {}

    // MODEL
    public static final String MODEL_INPUT_NAME = "input";
    public static final float ACCURACY_THRESHOLD = 0.60f; // lowered to match typical multiclass confidences

    // IMAGE
    public static final int IMAGE_WIDTH = 224;
    public static final int IMAGE_HEIGHT = 224;

    // FILE PATHS
    public static final String LABELS_FILE_PATH = "dataset/labels.txt"; // fallback - model/labels.txt preferred if present

    // UI
    public static final String APP_TITLE = "Non-Vehicle Detection System";
}
