package io.aortica.ecg.inference

/**
 * ECG signal preprocessing for the mobile inference pipeline.
 *
 * Handles resampling to the target sample rate (500 Hz) and
 * zero-padding/truncation to the expected input length.
 */
object SignalPreprocessor {

    /** Target sample rate for the ONNX edge model. */
    const val TARGET_SAMPLE_RATE = 500

    /** Target number of samples (10 seconds at 500 Hz). */
    const val TARGET_SAMPLES = 5000

    /**
     * Preprocess an ECG signal for inference.
     *
     * Performs:
     * 1. Resampling to [TARGET_SAMPLE_RATE] Hz if needed
     * 2. Padding or truncation to [TARGET_SAMPLES] samples
     * 3. Amplitude normalization (µV to mV if needed)
     *
     * @param signal Raw ECG signal, shape [numLeads, numSamples]
     * @param numLeads Number of leads in the signal
     * @param numSamples Number of samples per lead
     * @param sourceSampleRate Original sample rate of the signal in Hz
     * @param sourceUnits Units of the input signal: "uV" or "mV"
     * @return Preprocessed signal as flat FloatArray [numLeads * TARGET_SAMPLES]
     */
    fun preprocess(
        signal: FloatArray,
        numLeads: Int,
        numSamples: Int,
        sourceSampleRate: Int = TARGET_SAMPLE_RATE,
        sourceUnits: String = "mV"
    ): FloatArray {
        var processed = signal

        // Convert µV to mV if needed
        if (sourceUnits.equals("uV", ignoreCase = true) ||
            sourceUnits.equals("µV", ignoreCase = true)
        ) {
            processed = FloatArray(processed.size) { processed[it] / 1000f }
        }

        // Resample if needed
        if (sourceSampleRate != TARGET_SAMPLE_RATE) {
            processed = resample(processed, numLeads, numSamples, sourceSampleRate)
        }

        // Pad or truncate to TARGET_SAMPLES
        val currentSamples = if (sourceSampleRate != TARGET_SAMPLE_RATE) {
            (numSamples.toLong() * TARGET_SAMPLE_RATE / sourceSampleRate).toInt()
        } else {
            numSamples
        }

        processed = padOrTruncate(processed, numLeads, currentSamples)

        return processed
    }

    /**
     * Resample a multi-lead signal to the target sample rate using linear interpolation.
     *
     * @param signal Flat array of shape [numLeads * numSamples]
     * @param numLeads Number of leads
     * @param numSamples Current samples per lead
     * @param sourceSampleRate Source sample rate in Hz
     * @return Resampled signal as flat FloatArray
     */
    fun resample(
        signal: FloatArray,
        numLeads: Int,
        numSamples: Int,
        sourceSampleRate: Int
    ): FloatArray {
        val targetSamples = (numSamples.toLong() * TARGET_SAMPLE_RATE / sourceSampleRate).toInt()
        val result = FloatArray(numLeads * targetSamples)

        for (lead in 0 until numLeads) {
            val srcOffset = lead * numSamples
            val dstOffset = lead * targetSamples

            for (i in 0 until targetSamples) {
                val srcIdx = i.toFloat() * (numSamples - 1) / (targetSamples - 1).coerceAtLeast(1)
                val low = srcIdx.toInt().coerceIn(0, numSamples - 1)
                val high = (low + 1).coerceIn(0, numSamples - 1)
                val frac = srcIdx - low

                result[dstOffset + i] = signal[srcOffset + low] * (1f - frac) +
                    signal[srcOffset + high] * frac
            }
        }

        return result
    }

    /**
     * Pad (with zeros) or truncate a signal to [TARGET_SAMPLES] per lead.
     *
     * @param signal Flat array of shape [numLeads * currentSamples]
     * @param numLeads Number of leads
     * @param currentSamples Current samples per lead
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
            // Remaining samples are already zero (FloatArray default)
        }

        return result
    }
}
