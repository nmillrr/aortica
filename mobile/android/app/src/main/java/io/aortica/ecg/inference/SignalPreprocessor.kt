package io.aortica.ecg.inference

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * ECG signal preprocessing for the mobile inference pipeline.
 *
 * Produces exactly what the model was trained on — the contract defined in
 * `aortica/signal/preprocess.py`:
 *
 *  - **100 Hz**, **10 s** window (1,000 samples per lead), right zero-padded
 *    or truncated;
 *  - amplitudes in **micro-volts**;
 *  - no denoising.
 *
 * The ONNX graph accepts any signal length, so a 500 Hz input runs without
 * error and scores close to chance (macro-AUROC 0.53 vs 0.94 on PTB-XL
 * fold 10). An earlier version of this file fed 500 Hz, and converted
 * µV to mV — a further 1000x error. Change these constants only together
 * with the Python contract and a retrained model.
 */
object SignalPreprocessor {

    /** Sample rate the edge model was trained at. */
    const val TARGET_SAMPLE_RATE = 100

    /** Samples per lead (10 seconds at 100 Hz). */
    const val TARGET_SAMPLES = 1000

    /** Half-width of the anti-aliasing kernel, in output samples. */
    private const val KERNEL_HALF_WIDTH_OUT = 10

    /**
     * Preprocess an ECG signal for inference.
     *
     * @param signal Raw ECG signal, flat [numLeads * numSamples], lead-major
     * @param numLeads Number of leads in the signal
     * @param numSamples Number of samples per lead
     * @param sourceSampleRate Original sample rate of the signal in Hz
     * @param sourceUnits Units of the input signal: "mV", "uV"/"µV" or "V"
     * @return Flat FloatArray [numLeads * TARGET_SAMPLES] in µV at 100 Hz
     */
    fun preprocess(
        signal: FloatArray,
        numLeads: Int,
        numSamples: Int,
        sourceSampleRate: Int,
        sourceUnits: String = "mV"
    ): FloatArray {
        val scale = toMicrovoltsFactor(sourceUnits)
        var processed = if (scale == 1f) signal else FloatArray(signal.size) { signal[it] * scale }

        var currentSamples = numSamples
        if (sourceSampleRate != TARGET_SAMPLE_RATE) {
            processed = resample(processed, numLeads, numSamples, sourceSampleRate)
            currentSamples = resampledLength(numSamples, sourceSampleRate)
        }

        return padOrTruncate(processed, numLeads, currentSamples)
    }

    /** Multiplier converting [units] to µV. */
    fun toMicrovoltsFactor(units: String): Float = when (units.trim().lowercase()) {
        "uv", "µv", "μv" -> 1f
        "mv" -> 1_000f
        "v" -> 1_000_000f
        else -> throw IllegalArgumentException("Unrecognised signal units '$units'")
    }

    /** Samples per lead after resampling [numSamples] from [sourceSampleRate]. */
    fun resampledLength(numSamples: Int, sourceSampleRate: Int): Int =
        (numSamples.toDouble() * TARGET_SAMPLE_RATE / sourceSampleRate).roundToInt()

    /**
     * Resample a multi-lead signal to [TARGET_SAMPLE_RATE].
     *
     * Downsampling first low-passes with a zero-phase Hamming-windowed sinc
     * at the output Nyquist frequency, then interpolates; without that,
     * content above 50 Hz (powerline, EMG) aliases into the ECG band.
     * Upsampling interpolates linearly.
     *
     * @return Flat FloatArray [numLeads * resampledLength(numSamples, sourceSampleRate)]
     */
    fun resample(
        signal: FloatArray,
        numLeads: Int,
        numSamples: Int,
        sourceSampleRate: Int
    ): FloatArray {
        require(sourceSampleRate > 0) { "sourceSampleRate must be positive" }
        val targetSamples = resampledLength(numSamples, sourceSampleRate)
        val ratio = sourceSampleRate.toDouble() / TARGET_SAMPLE_RATE
        val kernel = if (ratio > 1.0) lowPassKernel(ratio) else null
        val result = FloatArray(numLeads * targetSamples)

        for (lead in 0 until numLeads) {
            val srcOffset = lead * numSamples
            val source = if (kernel != null) {
                convolveSame(signal, srcOffset, numSamples, kernel)
            } else {
                signal.copyOfRange(srcOffset, srcOffset + numSamples)
            }
            val dstOffset = lead * targetSamples
            for (i in 0 until targetSamples) {
                val srcIdx = i * ratio
                val low = srcIdx.toInt().coerceIn(0, numSamples - 1)
                val high = (low + 1).coerceIn(0, numSamples - 1)
                val frac = (srcIdx - low).toFloat()
                result[dstOffset + i] = source[low] * (1f - frac) + source[high] * frac
            }
        }
        return result
    }

    /** Hamming-windowed sinc with cutoff at the output Nyquist, unity DC gain. */
    internal fun lowPassKernel(ratio: Double): FloatArray {
        val half = (KERNEL_HALF_WIDTH_OUT * ratio).roundToInt()
        val cutoff = 0.5 / ratio // cycles per input sample
        val taps = DoubleArray(2 * half + 1) { k ->
            val n = (k - half).toDouble()
            val sinc = if (n == 0.0) 2 * cutoff else sin(2 * PI * cutoff * n) / (PI * n)
            val window = 0.54 - 0.46 * cos(2 * PI * k / (2 * half))
            sinc * window
        }
        val sum = taps.sum()
        return FloatArray(taps.size) { (taps[it] / sum).toFloat() }
    }

    /** Centred (zero-phase) convolution with edge samples held constant. */
    private fun convolveSame(
        signal: FloatArray,
        offset: Int,
        length: Int,
        kernel: FloatArray
    ): FloatArray {
        val half = kernel.size / 2
        val out = FloatArray(length)
        for (i in 0 until length) {
            var acc = 0f
            for (k in kernel.indices) {
                val j = (i + k - half).coerceIn(0, length - 1)
                acc += signal[offset + j] * kernel[k]
            }
            out[i] = acc
        }
        return out
    }

    /**
     * Right zero-pad or right-truncate to [TARGET_SAMPLES] per lead.
     *
     * @param signal Flat array of shape [numLeads * currentSamples]
     * @return Flat FloatArray of shape [numLeads * TARGET_SAMPLES]
     */
    fun padOrTruncate(
        signal: FloatArray,
        numLeads: Int,
        currentSamples: Int
    ): FloatArray {
        if (currentSamples == TARGET_SAMPLES) return signal

        val result = FloatArray(numLeads * TARGET_SAMPLES)
        val copyLen = minOf(currentSamples, TARGET_SAMPLES)
        for (lead in 0 until numLeads) {
            System.arraycopy(
                signal, lead * currentSamples,
                result, lead * TARGET_SAMPLES,
                copyLen
            )
        }
        return result
    }
}
