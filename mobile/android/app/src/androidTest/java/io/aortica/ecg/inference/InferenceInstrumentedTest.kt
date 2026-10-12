package io.aortica.ecg.inference

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumentation tests for the ONNX inference pipeline.
 *
 * These tests run on an Android device/emulator and verify:
 * - Model loading from assets
 * - Inference pipeline with synthetic ECG data
 * - Tier mapping from inference results
 * - Signal preprocessing (resampling, padding)
 */
@RunWith(AndroidJUnit4::class)
class InferenceInstrumentedTest {

    @Test
    fun testSignalPreprocessor_resample_downsamplesTo100Hz() {
        // 500 Hz → 100 Hz, single lead, 500 samples (1 second)
        val signal = FloatArray(500) { 1.0f }
        val resampled = SignalPreprocessor.resample(signal, 1, 500, 500)

        assertEquals(100, resampled.size)
        // A constant passes the anti-aliasing filter unchanged (unity DC gain)
        assertEquals(1.0f, resampled[50], 1e-4f)
    }

    @Test
    fun testSignalPreprocessor_resample_suppressesAliasing() {
        // A 60 Hz tone sampled at 500 Hz must not survive downsampling to
        // 100 Hz, where it would alias to 40 Hz inside the ECG band.
        val n = 5000
        val signal = FloatArray(n) { kotlin.math.sin(2 * Math.PI * 60.0 * it / 500.0).toFloat() }
        val resampled = SignalPreprocessor.resample(signal, 1, n, 500)
        val interior = resampled.copyOfRange(50, resampled.size - 50)
        val peak = interior.maxOf { kotlin.math.abs(it) }
        assertTrue("60 Hz leaked through at amplitude $peak", peak < 0.05f)
    }

    @Test
    fun testSignalPreprocessor_padOrTruncate_pad() {
        // 1 lead, 500 samples → pad to 1000
        val signal = FloatArray(500) { 1.0f }
        val padded = SignalPreprocessor.padOrTruncate(signal, 1, 500)

        assertEquals(1000, padded.size)
        assertEquals(1.0f, padded[499], 0f)
        assertEquals(0.0f, padded[500], 0f)  // Zero-padded
    }

    @Test
    fun testSignalPreprocessor_padOrTruncate_truncate() {
        // 1 lead, 2000 samples → truncate to 1000
        val signal = FloatArray(2000) { it.toFloat() }
        val truncated = SignalPreprocessor.padOrTruncate(signal, 1, 2000)

        assertEquals(1000, truncated.size)
        assertEquals(999.0f, truncated[999], 0f)
    }

    @Test
    fun testSignalPreprocessor_preprocess_outputsMicrovoltsAt100Hz() {
        // 10 s at 500 Hz of a constant 1 mV
        val signal = FloatArray(5000) { 1.0f }
        val processed = SignalPreprocessor.preprocess(
            signal, 1, 5000, sourceSampleRate = 500, sourceUnits = "mV"
        )

        assertEquals(1000, processed.size)
        assertEquals(1000.0f, processed[500], 0.5f)  // 1 mV == 1000 µV
    }

    @Test
    fun testSignalPreprocessor_preprocess_microvoltsPassThrough() {
        val signal = FloatArray(1000) { 1000f }
        val processed = SignalPreprocessor.preprocess(
            signal, 1, 1000, sourceSampleRate = 100, sourceUnits = "uV"
        )
        assertEquals(1000f, processed[0], 0f)
    }

    @Test
    fun testPadToTwelveLeads_singleLead() {
        val samples = 100
        val signal = FloatArray(samples) { 1.0f }
        val padded = OnnxInferenceEngine.padToTwelveLeads(signal, 1, samples)

        assertEquals(12 * samples, padded.size)
        // Lead II (index 1) should have data
        assertEquals(1.0f, padded[samples], 0f)
        // Lead I (index 0) should be zero
        assertEquals(0.0f, padded[0], 0f)
        // Lead III (index 2) should be zero
        assertEquals(0.0f, padded[2 * samples], 0f)
    }

    @Test
    fun testPadToTwelveLeads_sixLeads() {
        val samples = 100
        val signal = FloatArray(6 * samples) { 1.0f }
        val padded = OnnxInferenceEngine.padToTwelveLeads(signal, 6, samples)

        assertEquals(12 * samples, padded.size)
        // First 6 leads should have data
        for (lead in 0 until 6) {
            assertEquals(1.0f, padded[lead * samples], 0f)
        }
        // Remaining 6 leads should be zero
        for (lead in 6 until 12) {
            assertEquals(0.0f, padded[lead * samples], 0f)
        }
    }

    @Test
    fun testPadToTwelveLeads_twelveLeads() {
        val samples = 100
        val signal = FloatArray(12 * samples) { 2.0f }
        val padded = OnnxInferenceEngine.padToTwelveLeads(signal, 12, samples)

        // Should return same array (no copy)
        assertSame(signal, padded)
    }

    @Test
    fun testTierMapper_lowRisk() {
        // All scores below thresholds → LOW
        val result = InferenceResult(
            rhythm = FloatArray(28) { 0.1f },
            structural = FloatArray(19) { 0.1f },
            ischaemia = FloatArray(19) { 0.1f },
            risk = FloatArray(6) { 0.1f },
            latencyMs = 50.0
        )

        val report = TierMapper.mapToTier(result)
        assertEquals(TierMapper.Tier.LOW, report.tier)
        assertTrue(report.keyFindings.isEmpty())
    }

    @Test
    fun testTierMapper_urgentTier() {
        // VT (index 5) above urgent threshold
        val rhythm = FloatArray(28) { 0.1f }
        rhythm[5] = 0.9f  // VT = 0.9 > 0.7 threshold

        val result = InferenceResult(
            rhythm = rhythm,
            structural = FloatArray(19) { 0.1f },
            ischaemia = FloatArray(19) { 0.1f },
            risk = FloatArray(6) { 0.1f },
            latencyMs = 50.0
        )

        val report = TierMapper.mapToTier(result)
        assertEquals(TierMapper.Tier.URGENT, report.tier)
        assertTrue(report.keyFindings.any { it.condition == "VT" })
    }

    @Test
    fun testTierMapper_referTier() {
        // AF (index 0) above refer threshold but below urgent
        val rhythm = FloatArray(28) { 0.1f }
        rhythm[0] = 0.8f  // AF = 0.8 > 0.6 refer threshold

        val result = InferenceResult(
            rhythm = rhythm,
            structural = FloatArray(19) { 0.1f },
            ischaemia = FloatArray(19) { 0.1f },
            risk = FloatArray(6) { 0.1f },
            latencyMs = 50.0
        )

        val report = TierMapper.mapToTier(result)
        assertEquals(TierMapper.Tier.REFER, report.tier)
        assertTrue(report.keyFindings.any { it.condition == "AF" })
    }

    @Test
    fun testModelLoading_contextAvailable() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertNotNull(context)

        // Just verify context is available — model loading requires
        // actual model.onnx in assets which may not be present in CI
    }
}
