package com.nvd.camera;

import java.awt.image.BufferedImage;
import java.lang.reflect.Method;
import java.lang.reflect.Field;

import com.nvd.exception.CameraAccessException;

/**
 * CameraService attempts to use the system webcam when a runtime
 * webcam library is present (sarxos webcam-capture). If not,
 * it falls back to a simulated camera.
 */
public class CameraService {

    private boolean active = false;
    private BufferedImage lastFrame = null;

    // runtime webcam object (from sarxos webcam-capture) held by reflection
    private Object webcam = null;
    private boolean useWebcamLib = false;

    // optionally loaded class and loader
    private Class<?> webcamClass = null;
    private java.net.URLClassLoader externalLoader = null;

    public CameraService() {
        initializeCamera();
    }

    /**
     * Initialize camera: detect if sarxos webcam library is available at runtime
     * If Class.forName fails, attempt to locate a JAR in ./lib and load it dynamically.
     */
    private void initializeCamera() {
        try {
            webcamClass = Class.forName("com.github.sarxos.webcam.Webcam");
            useWebcamLib = true;
            System.out.println("Webcam library detected on classpath; system camera may be available.");
            return;
        } catch (ClassNotFoundException e) {
            // try to locate jar in local lib folder
            try {
                String[] candidateDirs = new String[] {"lib", "NonVehicleDetectionSystem/lib", "../NonVehicleDetectionSystem/lib", "./NonVehicleDetectionSystem/lib"};
                for (String dir : candidateDirs) {
                    java.io.File libDir = new java.io.File(dir);
                    if (libDir.exists() && libDir.isDirectory()) {
                        java.io.File[] matches = libDir.listFiles((d, n) -> n.toLowerCase().endsWith(".jar"));
                        if (matches != null && matches.length > 0) {
                            java.net.URL[] urls = new java.net.URL[matches.length];
                            for (int i = 0; i < matches.length; i++) urls[i] = matches[i].toURI().toURL();
                            externalLoader = new java.net.URLClassLoader(urls, getClass().getClassLoader());
                            // Try to load Webcam class from the combined classloader
                            try {
                                webcamClass = externalLoader.loadClass("com.github.sarxos.webcam.Webcam");
                                useWebcamLib = true;
                                System.out.println("Webcam library loaded (with dependencies) from " + libDir.getAbsolutePath() + "; system camera may be available.");
                                return;
                            } catch (ClassNotFoundException cnf) {
                                // continue searching other dirs
                            }
                        }
                    }
                }
            } catch (Exception ex) {
                System.err.println("Failed to dynamically load webcam library: " + ex.getMessage());
            }

            System.out.println("No runtime webcam library detected; camera will be simulated.");
            useWebcamLib = false;
        }
    }

    /**
     * Start camera
     */
    public void startCamera() throws CameraAccessException {
        try {
            if (useWebcamLib && webcamClass != null) {
                // Open default webcam via reflection on the loaded class
                Method getDefault = webcamClass.getMethod("getDefault");
                webcam = getDefault.invoke(null);
                if (webcam == null) {
                    System.out.println("No default webcam found; falling back to simulated camera.");
                    useWebcamLib = false;
                } else {
                    Method open = webcamClass.getMethod("open");
                    open.invoke(webcam);
                    System.out.println("System webcam opened (via webcam-capture library).");
                }
            }

            active = true;
            if (!useWebcamLib) {
                System.out.println("Camera started (simulated)");
            }

        } catch (Exception e) {
            throw new CameraAccessException("Unable to access camera: " + e.getMessage(), e);
        }
    }

    /**
     * Stop camera
     */
    public void stopCamera() {
        try {
            if (webcam != null) {
                Method close = webcam.getClass().getMethod("close");
                close.invoke(webcam);
                webcam = null;
                System.out.println("System webcam closed.");
            }
        } catch (Exception ignored) {
        }
        active = false;
        System.out.println("Camera stopped");
    }

    /**
     * Read next frame (returns BufferedImage)
     */
    public BufferedImage getFrame() {
        if (!active) return null;

        if (useWebcamLib && webcam != null) {
            try {
                Method getImage = webcam.getClass().getMethod("getImage");
                Object img = getImage.invoke(webcam);
                if (img instanceof BufferedImage) {
                    lastFrame = (BufferedImage) img;
                }
            } catch (Exception e) {
                System.err.println("Failed to capture image from webcam: " + e.getMessage());
                useWebcamLib = false; // disable for this session
            }
        }

        if (lastFrame == null) {
            lastFrame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        }
        return lastFrame;
    }

    public boolean isActive() {
        return active;
    }
}
