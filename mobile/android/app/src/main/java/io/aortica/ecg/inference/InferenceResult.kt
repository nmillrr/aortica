package io.aortica.ecg.inference

/**
 * Raw multi-task inference output from the ONNX edge model.
 *
 * Each array contains sigmoid-activated probabilities for the corresponding
 * task head's classes. The [risk] array contains regression scores (0–1).
 *
 * @property rhythm Rhythm classification probabilities (28 classes).
 * @property structural Structural classification probabilities (19 classes).
 * @property ischaemia Ischaemia classification probabilities (19 classes).
 * @property risk Risk regression scores (6 outputs).
 * @property latencyMs Inference latency in milliseconds.
 */
data class InferenceResult(
    val rhythm: FloatArray,
    val structural: FloatArray,
    val ischaemia: FloatArray,
    val risk: FloatArray,
    val latencyMs: Double
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is InferenceResult) return false
        return rhythm.contentEquals(other.rhythm) &&
            structural.contentEquals(other.structural) &&
            ischaemia.contentEquals(other.ischaemia) &&
            risk.contentEquals(other.risk) &&
            latencyMs == other.latencyMs
    }

    override fun hashCode(): Int {
        var result = rhythm.contentHashCode()
        result = 31 * result + structural.contentHashCode()
        result = 31 * result + ischaemia.contentHashCode()
        result = 31 * result + risk.contentHashCode()
        result = 31 * result + latencyMs.hashCode()
        return result
    }
}
