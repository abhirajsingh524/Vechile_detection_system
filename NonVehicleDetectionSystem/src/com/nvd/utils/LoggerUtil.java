package com.nvd.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Logger utility for application logging
 */
public class LoggerUtil {

    private static final String LOG_FILE = "output/logs/detection.log";
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    private LoggerUtil() {
        // Prevent instantiation
    }

    /**
     * Log information message
     */
    public static void info(String message) {
        log("INFO", message);
    }

    /**
     * Log warning message
     */
    public static void warn(String message) {
        log("WARN", message);
    }

    /**
     * Log error message
     */
    public static void error(String message) {
        log("ERROR", message);
    }

    /**
     * Log error with exception
     */
    public static void error(String message, Throwable throwable) {
        log("ERROR", message + " - " + throwable.getMessage());
        throwable.printStackTrace();
    }

    /**
     * Log debug message
     */
    public static void debug(String message) {
        log("DEBUG", message);
    }

    /**
     * Internal log method
     */
    private static void log(String level, String message) {
        String timestamp = DATE_FORMAT.format(new Date());
        String logEntry = String.format("[%s] %s - %s", timestamp, level, message);
        
        // Print to console
        System.out.println(logEntry);
        
        // Write to file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(LOG_FILE, true))) {
            writer.write(logEntry);
            writer.newLine();
        } catch (IOException e) {
            System.err.println("Failed to write to log file: " + e.getMessage());
        }
    }
}
