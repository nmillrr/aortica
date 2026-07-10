package io.aortica.ecg.inference

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * ONNX Runtime Mobile inference engine for ECG analysis.
 *
 * Loads the INT8 quantized edge model from the app's assets directory
 * and runs multi-task inference (rhythm, structural, ischaemia, risk)
 * on preprocessed ECG signals.
 *
 * Thread-safe: all inference runs on [Dispatchers.Default] via coroutines.
 *
 * @param context Application context for asset loading.
 * @param modelFileName Name of the ONNX model file in assets/. Default: "model.onnx"
 */
class OnnxInferenceEngine(
    private val context: Context,
    private val modelFileName: String = MODEL_FILENAME
) : Closeable {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val sessionLock = Any()

    /**
     * Ensure the ONNX session is loaded. Idempotent — safe to call multiple times.
     */
    fun loadModel() {
        synchronized(sessionLock) {
            if (session != null) return
            val modelBytes = context.assets.open(modelFileName).use { it.readBytes() }
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            session = ortEnv.createSession(modelBytes, sessionOptions)
        }
    }

    /**
     * Check if the model is currently loaded.
     */
    val isModelLoaded: Boolean
        get() = synchronized(sessionLock) { session != null }

    /**
     * Run inference on a preprocessed ECG signal.
     *
     * @param signal ECG signal as a flat FloatArray with shape [1, leads, samples].
     *               For single-lead: [1, 1, samples] (will be zero-padded to 12 leads).
     *               For 6-lead limb: [1, 6, samples] (will be zero-padded to 12 leads).
     *               For 12-lead: [1, 12, samples] (used directly).
     * @param numLeads Number of leads in the input signal (1, 6, or 12).
     * @param numSamples Number of samples per lead.
     * @return [InferenceResult] with multi-task predictions.
     */
    suspend fun predict(
        signal: FloatArray,
        numLeads: Int,
        numSamples: Int
    ): InferenceResult = withContext(Dispatchers.Default) {
        loadModel()

        val startTime = System.nanoTime()

        // Zero-pad to 12 leads if needed
        val paddedSignal = padToTwelveLeads(signal, numLeads, numSamples)

        // Create ONNX tensor: shape [1, 12, numSamples]
        val shape = longArrayOf(1, 12, numSamples.toLong())
        val buffer = FloatBuffer.wrap(paddedSignal)
        val inputTensor = OnnxTensor.createTensor(ortEnv, buffer, shape)

        try {
            val session = synchronized(sessionLock) {
                this@OnnxInferenceEngine.session
                    ?: throw IllegalStateException("Model not loaded")
            }

            val inputName = session.inputNames.first()
            val results = session.run(mapOf(inputName to inputTensor))

            val outputs = mutableMapOf<String, FloatArray>()
            for ((i, outputInfo) in session.outputNames.withIndex()) {
                val tensor = results.get(i).orElse(null)
                if (tensor is OnnxTensor) {
                    outputs[outputInfo] = tensor.floatBuffer.let { buf ->
                        FloatArray(buf.remaining()).also { buf.get(it) }
                    }
                }
            }

            val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0

            InferenceResult(
                rhythm = outputs["rhythm_output"] ?: FloatArray(0),
                structural = outputs["structural_output"] ?: FloatArray(0),
                ischaemia = outputs["ischaemia_output"] ?: FloatArray(0),
                risk = outputs["risk_output"] ?: FloatArray(0),
                latencyMs = latencyMs
            )
        } finally {
            inputTensor.close()
        }
    }

    /**
     * Release the ONNX session and free native resources.
     */
    override fun close() {
        synchronized(sessionLock) {
            session?.close()
            session = null
        }
    }

    companion object {
        /** Default model filename in assets/ directory. */
        const val MODEL_FILENAME = "model.onnx"

        /** Target sample rate for the edge model. */
        const val TARGET_SAMPLE_RATE = 500

        /** Standard 12-lead count. */
        const val STANDARD_LEAD_COUNT = 12

        /** Maximum supported samples (10 seconds at 500 Hz). */
        const val MAX_SAMPLES = 5000

        /**
         * Zero-pad a signal to 12 leads.
         *
         * Supports single-lead (Lead II placed at index 1),
         * 6-lead limb (placed at indices 0–5), and 12-lead (pass-through).
         *
         * @param signal Flat float array of shape [1, numLeads, numSamples]
         * @param numLeads Number of actual leads (1, 6, or 12)
         * @param numSamples Samples per lead
         * @return Flat float array of shape [1, 12, numSamples]
         */
        fun padToTwelveLeads(
            signal: FloatArray,
            numLeads: Int,
            numSamples: Int
        ): FloatArray {
            if (numLeads == STANDARD_LEAD_COUNT) return signal

            val padded = FloatArray(STANDARD_LEAD_COUNT * numSamples)

            when (numLeads) {
                1 -> {
                    // Single lead → place at Lead II (index 1)
                    System.arraycopy(signal, 0, padded, numSamples, numSamples)
                }
                6 -> {
                    // 6-lead limb → place at indices 0–5 (I, II, III, aVR, aVL, aVF)
                    System.arraycopy(signal, 0, padded, 0, 6 * numSamples)
                }
                else -> {
                    // Copy whatever leads are available
                    val copyLeads = minOf(numLeads, STANDARD_LEAD_COUNT)
                    System.arraycopy(signal, 0, padded, 0, copyLeads * numSamples)
                }
            }

            return padded
        }
    }
}
