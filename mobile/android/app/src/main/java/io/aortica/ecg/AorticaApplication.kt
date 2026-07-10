package io.aortica.ecg

import android.app.Application
import io.aortica.ecg.audit.AuditLogger
import io.aortica.ecg.inference.OnnxInferenceEngine

/**
 * Application-level initialization for the Aortica ECG app.
 *
 * Initializes the ONNX inference engine and audit logger on startup.
 * Both components are lazily initialized and available throughout
 * the application lifecycle.
 */
class AorticaApplication : Application() {

    /** ONNX inference engine — lazily initialized on first access. */
    val inferenceEngine: OnnxInferenceEngine by lazy {
        OnnxInferenceEngine(this)
    }

    /** Audit logger for recording inference events. */
    val auditLogger: AuditLogger by lazy {
        AuditLogger(this)
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        /** Singleton application instance for global access. */
        @Volatile
        lateinit var instance: AorticaApplication
            private set
    }
}
