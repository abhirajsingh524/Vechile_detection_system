Non-Vehicle Detection System
=========================

Java Swing application for detecting non-vehicles using ONNX ML model

---

Quick setup notes:

- To enable the system webcam capture feature, either:
  - Run the PowerShell helper (no admin required) to download the sarxos webcam-capture JAR into `lib/`:

      powershell -ExecutionPolicy Bypass -File scripts\download_webcam_jar.ps1

  - Or use Maven to fetch dependencies (dependency already added to `pom.xml`).

- Exporting a PyTorch model to ONNX (example):

  python tools\export_to_onnx.py --checkpoint path\to\DriveMind.pth --output model\vehicle_detector.onnx --num-classes 2

- Run the app with the system camera (after adding the webcam JAR):

  java -cp "target/classes;lib/*" com.nvd.main.MainApp camera

- Run detection on a specific image:

  java -cp "target/classes;lib/*" com.nvd.main.MainApp path\to\image.jpg
