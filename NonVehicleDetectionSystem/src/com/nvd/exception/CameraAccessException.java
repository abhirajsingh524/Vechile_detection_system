package com.nvd.exception;

public class CameraAccessException extends Exception {

    public CameraAccessException(String message) {
        super(message);
    }

    public CameraAccessException(String message, Throwable cause) {
        super(message, cause);
    }
}
