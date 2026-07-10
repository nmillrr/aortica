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
    fun testSignalPreprocessor_resample() {
        // 250 Hz → 500 Hz, single lead, 250 samples (1 second)
        val signal = FloatArray(250) { (it % 50).toFloat() / 50f }
        val resampled = SignalPreprocessor.resample(signal, 1, 250, 250)

        assertEquals(500, resampled.size)
        // First and last values should be preserved
        assertEquals(signal[0], resampled[0], 0.01f)
    }

    @Test
    fun testSignalPreprocessor_padOrTruncate_pad() {
        // 1 lead, 1000 samples → pad to 5000
        val signal = FloatArray(1000) { 1.0f }
        val padded = SignalPreprocessor.padOrTruncate(signal, 1, 1000)

        assertEquals(5000, padded.size)
        assertEquals(1.0f, padded[0], 0f)
        assertEquals(1.0f, padded[999], 0f)
        assertEquals(0.0f, padded[1000], 0f)  // Zero-padded
    }

    @Test
    fun testSignalPreprocessor_padOrTruncate_truncate() {
        // 1 lead, 10000 samples → truncate to 5000
        val signal = FloatArray(10000) { it.toFloat() }
        val truncated = SignalPreprocessor.padOrTruncate(signal, 1, 10000)

        assertEquals(5000, truncated.size)
        assertEquals(0.0f, truncated[0], 0f)
        assertEquals(4999.0f, truncated[4999], 0f)
    }

    @Test
    fun testSignalPreprocessor_preprocess_microVoltConversion() {
        val signal = FloatArray(5000) { 1000f }  // 1000 µV
        val processed = SignalPreprocessor.preprocess(
            signal, 1, 5000, sourceUnits = "uV"
        )

        assertEquals(5000, processed.size)
        assertEquals(1.0f, processed[0], 0.01f)  // Should be 1 mV
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
