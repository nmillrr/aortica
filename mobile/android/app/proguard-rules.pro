# Aortica ECG Android ProGuard Rules

# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# Keep Kotlin metadata
-keepattributes *Annotation*
-keep class kotlin.Metadata { *; }

# Keep data classes used for JSON serialization
-keep class io.aortica.ecg.inference.** { *; }
-keep class io.aortica.ecg.audit.** { *; }
-keep class io.aortica.ecg.model.** { *; }
