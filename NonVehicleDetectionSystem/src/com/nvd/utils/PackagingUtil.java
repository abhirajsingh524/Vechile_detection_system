package com.nvd.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class PackagingUtil {

    private PackagingUtil() {}

    /**
     * Export model files into a ZIP. Includes original labels and a generic labels file
     */
    public static File exportModelPackage(File outFile) throws IOException {
        // Try candidate model directories
        String[] candidates = new String[] {"model", "NonVehicleDetectionSystem/model", "../model"};
        File modelDir = null;
        for (String c : candidates) {
            File d = new File(c);
            if (d.exists() && d.isDirectory()) {
                File mf = new File(d, "vehicle_detector.onnx");
                if (mf.exists()) {
                    modelDir = d;
                    break;
                }
            }
        }
        if (modelDir == null) throw new IOException("Model directory not found in candidates");

        File modelFile = new File(modelDir, "vehicle_detector.onnx");
        File labelsFile = new File(modelDir, "labels.txt");
        if (!modelFile.exists()) throw new IOException("Model file not found: " + modelFile.getAbsolutePath());

        File outDir = outFile.getParentFile();
        if (outDir != null) outDir.mkdirs();

        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(outFile))) {
            addFileToZip(zos, modelFile, "model/vehicle_detector.onnx");

            // include original labels if present
            if (labelsFile.exists()) {
                addFileToZip(zos, labelsFile, "model/labels.txt");

                // create a generic labels entry (vehicle / non_vehicle)
                String generic = generateGenericLabelsContent(labelsFile);
                ZipEntry gen = new ZipEntry("model/labels_generic.txt");
                zos.putNextEntry(gen);
                zos.write(generic.getBytes());
                zos.closeEntry();
            }
        }
        return outFile;
    }

    private static void addFileToZip(ZipOutputStream zos, File file, String entryName) throws IOException {
        ZipEntry entry = new ZipEntry(entryName);
        zos.putNextEntry(entry);
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] buffer = new byte[4096];
            int len;
            while ((len = fis.read(buffer)) > 0) zos.write(buffer, 0, len);
        }
        zos.closeEntry();
    }

    private static String generateGenericLabelsContent(File labelsFile) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (String line : Files.readAllLines(labelsFile.toPath())) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) continue;
            String gen = "vehicle";
            if (trimmed.equalsIgnoreCase("non_vehicle")) gen = "non_vehicle";
            sb.append(gen).append('\n');
        }
        return sb.toString();
    }
}