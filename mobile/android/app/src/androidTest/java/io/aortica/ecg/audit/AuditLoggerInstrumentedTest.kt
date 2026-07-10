package io.aortica.ecg.audit

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aortica.ecg.inference.TierMapper
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumentation tests for the audit logger.
 *
 * Tests CRUD operations, sync queue behavior, and anonymization.
 */
@RunWith(AndroidJUnit4::class)
class AuditLoggerInstrumentedTest {

    private lateinit var auditLogger: AuditLogger

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // Use a test database
        context.deleteDatabase("test_audit.db")
        auditLogger = AuditLogger(context)
    }

    @After
    fun tearDown() {
        auditLogger.clearAll()
        auditLogger.close()
    }

    @Test
    fun testLogInference_insertsEntry() {
        val signal = FloatArray(100) { it.toFloat() }

        val rowId = auditLogger.logInference(
            inputSignal = signal,
            tier = TierMapper.Tier.LOW,
            latencyMs = 42.5,
            numLeads = 1,
            numSamples = 100
        )

        assertTrue("Row should be inserted", rowId > 0)
        assertEquals(1, auditLogger.totalCount())
    }

    @Test
    fun testLogInference_createsHashNotRawData() {
        val signal = FloatArray(100) { it.toFloat() }

        auditLogger.logInference(
            inputSignal = signal,
            tier = TierMapper.Tier.REFER,
            latencyMs = 100.0,
            numLeads = 6,
            numSamples = 100
        )

        val entries = auditLogger.getEntries()
        assertEquals(1, entries.size)

        val entry = entries[0]
        val hash = entry.getString("input_hash")
        // Hash should be a hex string (64 chars for SHA-256)
        assertEquals(64, hash.length)
        assertTrue(hash.matches(Regex("[0-9a-f]+")))

        // Tier should be stored
        assertEquals("refer", entry.getString("tier"))
    }

    @Test
    fun testLogInference_multipleEntries() {
        val signal = FloatArray(50) { 1.0f }

        repeat(5) {
            auditLogger.logInference(signal, TierMapper.Tier.LOW, 10.0, 1, 50)
        }

        assertEquals(5, auditLogger.totalCount())
        assertEquals(5, auditLogger.pendingCount())
    }

    @Test
    fun testGetEntries_unSyncedOnly() {
        val signal = FloatArray(50) { 1.0f }

        val id1 = auditLogger.logInference(signal, TierMapper.Tier.LOW, 10.0, 1, 50)
        auditLogger.logInference(signal, TierMapper.Tier.URGENT, 20.0, 1, 50)

        // Mark first as synced
        auditLogger.markSynced(listOf(id1))

        val unSynced = auditLogger.getEntries(unSyncedOnly = true)
        assertEquals(1, unSynced.size)
        assertEquals("urgent", unSynced[0].getString("tier"))
    }

    @Test
    fun testPendingSyncBatch_returnsCorrectEntries() {
        val signal = FloatArray(50) { 1.0f }

        auditLogger.logInference(signal, TierMapper.Tier.LOW, 10.0, 1, 50)
        auditLogger.logInference(signal, TierMapper.Tier.REFER, 20.0, 6, 50)

        val batch = auditLogger.getPendingSyncBatch(10)
        assertEquals(2, batch.length())

        // Verify anonymization: no raw signal data in batch
        for (i in 0 until batch.length()) {
            val entry = batch.getJSONObject(i)
            assertTrue(entry.has("input_hash"))
            assertTrue(entry.has("tier"))
            assertTrue(entry.has("latency_ms"))
            assertFalse(entry.has("signal"))
            assertFalse(entry.has("raw_data"))
        }
    }

    @Test
    fun testMarkSynced_removesFromQueue() {
        val signal = FloatArray(50) { 1.0f }

        val id1 = auditLogger.logInference(signal, TierMapper.Tier.LOW, 10.0, 1, 50)
        val id2 = auditLogger.logInference(signal, TierMapper.Tier.REFER, 20.0, 1, 50)

        assertEquals(2, auditLogger.pendingCount())

        auditLogger.markSynced(listOf(id1))
        assertEquals(1, auditLogger.pendingCount())

        auditLogger.markSynced(listOf(id2))
        assertEquals(0, auditLogger.pendingCount())
    }

    @Test
    fun testClearAll_removesEverything() {
        val signal = FloatArray(50) { 1.0f }

        repeat(3) {
            auditLogger.logInference(signal, TierMapper.Tier.LOW, 10.0, 1, 50)
        }

        assertEquals(3, auditLogger.totalCount())
        auditLogger.clearAll()
        assertEquals(0, auditLogger.totalCount())
        assertEquals(0, auditLogger.pendingCount())
    }

    @Test
    fun testHashSignal_deterministic() {
        val signal = FloatArray(100) { it.toFloat() }

        val hash1 = AuditLogger.hashSignal(signal)
        val hash2 = AuditLogger.hashSignal(signal)

        assertEquals(hash1, hash2)
    }

    @Test
    fun testHashSignal_differentForDifferentSignals() {
        val signal1 = FloatArray(100) { it.toFloat() }
        val signal2 = FloatArray(100) { (it + 1).toFloat() }

        val hash1 = AuditLogger.hashSignal(signal1)
        val hash2 = AuditLogger.hashSignal(signal2)

        assertNotEquals(hash1, hash2)
    }

    @Test
    fun testIsoTimestamp_format() {
        val ts = AuditLogger.isoTimestamp()
        assertTrue("Timestamp should be ISO format", ts.matches(Regex("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z")))
    }
}
