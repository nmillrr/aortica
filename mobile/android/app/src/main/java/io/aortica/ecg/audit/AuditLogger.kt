package io.aortica.ecg.audit

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.content.ContentValues
import io.aortica.ecg.inference.TierMapper
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Anonymized audit log for recording inference events.
 *
 * Stores each inference (timestamp, input hash, tier result, inference latency)
 * in local SQLite. Supports syncing to a remote endpoint when connectivity
 * is detected, stripping all patient-identifiable metadata before upload.
 *
 * Reuses the sync protocol from US-055/US-056.
 */
class AuditLogger(context: Context) : SQLiteOpenHelper(
    context, DATABASE_NAME, null, DATABASE_VERSION
) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS $TABLE_AUDIT_LOG (
                $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_TIMESTAMP TEXT NOT NULL,
                $COL_INPUT_HASH TEXT NOT NULL,
                $COL_TIER TEXT NOT NULL,
                $COL_LATENCY_MS REAL NOT NULL,
                $COL_NUM_LEADS INTEGER NOT NULL,
                $COL_NUM_SAMPLES INTEGER NOT NULL,
                $COL_DEVICE_ID TEXT,
                $COL_SYNCED INTEGER DEFAULT 0,
                $COL_SYNC_TIMESTAMP TEXT
            )
        """.trimIndent())

        db.execSQL("""
            CREATE INDEX IF NOT EXISTS idx_synced ON $TABLE_AUDIT_LOG ($COL_SYNCED)
        """.trimIndent())

        db.execSQL("""
            CREATE TABLE IF NOT EXISTS $TABLE_SYNC_QUEUE (
                $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_AUDIT_ID INTEGER NOT NULL,
                $COL_CREATED_AT TEXT NOT NULL,
                $COL_RETRY_COUNT INTEGER DEFAULT 0,
                FOREIGN KEY ($COL_AUDIT_ID) REFERENCES $TABLE_AUDIT_LOG($COL_ID)
            )
        """.trimIndent())
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Simple migration: recreate tables
        db.execSQL("DROP TABLE IF EXISTS $TABLE_SYNC_QUEUE")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_AUDIT_LOG")
        onCreate(db)
    }

    /**
     * Record an inference event in the audit log.
     *
     * @param inputSignal The raw input signal (used only for hashing — NOT stored)
     * @param tier The assigned risk tier
     * @param latencyMs Inference latency in milliseconds
     * @param numLeads Number of leads in the input
     * @param numSamples Number of samples per lead
     * @param deviceId Optional device identifier
     * @return The row ID of the inserted audit log entry
     */
    fun logInference(
        inputSignal: FloatArray,
        tier: TierMapper.Tier,
        latencyMs: Double,
        numLeads: Int,
        numSamples: Int,
        deviceId: String? = null
    ): Long {
        val db = writableDatabase
        val timestamp = isoTimestamp()
        val inputHash = hashSignal(inputSignal)

        val values = ContentValues().apply {
            put(COL_TIMESTAMP, timestamp)
            put(COL_INPUT_HASH, inputHash)
            put(COL_TIER, tier.key)
            put(COL_LATENCY_MS, latencyMs)
            put(COL_NUM_LEADS, numLeads)
            put(COL_NUM_SAMPLES, numSamples)
            put(COL_DEVICE_ID, deviceId)
            put(COL_SYNCED, 0)
        }

        val rowId = db.insert(TABLE_AUDIT_LOG, null, values)

        // Queue for sync
        if (rowId != -1L) {
            val queueValues = ContentValues().apply {
                put(COL_AUDIT_ID, rowId)
                put(COL_CREATED_AT, timestamp)
            }
            db.insert(TABLE_SYNC_QUEUE, null, queueValues)
        }

        return rowId
    }

    /**
     * Get all audit log entries.
     *
     * @param limit Maximum number of entries to return (default 100)
     * @param unSyncedOnly If true, return only entries not yet synced
     * @return List of audit log entries as JSON objects
     */
    fun getEntries(limit: Int = 100, unSyncedOnly: Boolean = false): List<JSONObject> {
        val db = readableDatabase
        val selection = if (unSyncedOnly) "$COL_SYNCED = 0" else null
        val cursor = db.query(
            TABLE_AUDIT_LOG,
            null,
            selection,
            null, null, null,
            "$COL_ID DESC",
            limit.toString()
        )

        val entries = mutableListOf<JSONObject>()
        cursor.use {
            while (it.moveToNext()) {
                entries.add(JSONObject().apply {
                    put("id", it.getLong(it.getColumnIndexOrThrow(COL_ID)))
                    put("timestamp", it.getString(it.getColumnIndexOrThrow(COL_TIMESTAMP)))
                    put("input_hash", it.getString(it.getColumnIndexOrThrow(COL_INPUT_HASH)))
                    put("tier", it.getString(it.getColumnIndexOrThrow(COL_TIER)))
                    put("latency_ms", it.getDouble(it.getColumnIndexOrThrow(COL_LATENCY_MS)))
                    put("num_leads", it.getInt(it.getColumnIndexOrThrow(COL_NUM_LEADS)))
                    put("num_samples", it.getInt(it.getColumnIndexOrThrow(COL_NUM_SAMPLES)))
                    put("device_id", it.getString(it.getColumnIndexOrThrow(COL_DEVICE_ID)))
                    put("synced", it.getInt(it.getColumnIndexOrThrow(COL_SYNCED)) == 1)
                })
            }
        }

        return entries
    }

    /**
     * Get the count of pending (un-synced) audit entries.
     */
    fun pendingCount(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery(
            "SELECT COUNT(*) FROM $TABLE_SYNC_QUEUE", null
        )
        cursor.use {
            return if (it.moveToFirst()) it.getInt(0) else 0
        }
    }

    /**
     * Get entries pending sync, anonymized for upload.
     *
     * Strips all patient-identifiable metadata before returning.
     * Only includes: timestamp, input_hash, tier, latency_ms, num_leads, device_id.
     *
     * @param batchSize Maximum entries per sync batch
     * @return JSON array of anonymized entries ready for upload
     */
    fun getPendingSyncBatch(batchSize: Int = 50): JSONArray {
        val db = readableDatabase
        val cursor = db.rawQuery("""
            SELECT a.$COL_ID, a.$COL_TIMESTAMP, a.$COL_INPUT_HASH,
                   a.$COL_TIER, a.$COL_LATENCY_MS, a.$COL_NUM_LEADS,
                   a.$COL_DEVICE_ID
            FROM $TABLE_AUDIT_LOG a
            INNER JOIN $TABLE_SYNC_QUEUE q ON a.$COL_ID = q.$COL_AUDIT_ID
            ORDER BY a.$COL_ID ASC
            LIMIT ?
        """.trimIndent(), arrayOf(batchSize.toString()))

        val batch = JSONArray()
        cursor.use {
            while (it.moveToNext()) {
                batch.put(JSONObject().apply {
                    put("id", it.getLong(0))
                    put("timestamp", it.getString(1))
                    put("input_hash", it.getString(2))
                    put("tier", it.getString(3))
                    put("latency_ms", it.getDouble(4))
                    put("num_leads", it.getInt(5))
                    // device_id is anonymized — included for site-level analytics
                    put("device_id", it.getString(6))
                })
            }
        }

        return batch
    }

    /**
     * Mark entries as synced after successful upload.
     *
     * @param ids List of audit log entry IDs that were successfully synced
     */
    fun markSynced(ids: List<Long>) {
        if (ids.isEmpty()) return

        val db = writableDatabase
        val timestamp = isoTimestamp()
        db.beginTransaction()
        try {
            for (id in ids) {
                db.execSQL(
                    "UPDATE $TABLE_AUDIT_LOG SET $COL_SYNCED = 1, $COL_SYNC_TIMESTAMP = ? WHERE $COL_ID = ?",
                    arrayOf(timestamp, id)
                )
                db.execSQL(
                    "DELETE FROM $TABLE_SYNC_QUEUE WHERE $COL_AUDIT_ID = ?",
                    arrayOf(id)
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Get total count of audit log entries.
     */
    fun totalCount(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT COUNT(*) FROM $TABLE_AUDIT_LOG", null)
        cursor.use {
            return if (it.moveToFirst()) it.getInt(0) else 0
        }
    }

    /**
     * Delete all audit log entries and sync queue.
     */
    fun clearAll() {
        val db = writableDatabase
        db.delete(TABLE_SYNC_QUEUE, null, null)
        db.delete(TABLE_AUDIT_LOG, null, null)
    }

    companion object {
        const val DATABASE_NAME = "aortica_audit.db"
        const val DATABASE_VERSION = 1

        const val TABLE_AUDIT_LOG = "audit_log"
        const val TABLE_SYNC_QUEUE = "sync_queue"

        const val COL_ID = "_id"
        const val COL_TIMESTAMP = "timestamp"
        const val COL_INPUT_HASH = "input_hash"
        const val COL_TIER = "tier"
        const val COL_LATENCY_MS = "latency_ms"
        const val COL_NUM_LEADS = "num_leads"
        const val COL_NUM_SAMPLES = "num_samples"
        const val COL_DEVICE_ID = "device_id"
        const val COL_SYNCED = "synced"
        const val COL_SYNC_TIMESTAMP = "sync_timestamp"
        const val COL_AUDIT_ID = "audit_id"
        const val COL_CREATED_AT = "created_at"
        const val COL_RETRY_COUNT = "retry_count"

        /**
         * Hash a signal array using SHA-256 for audit trail.
         * The hash is computed over the raw float bytes — no patient data is stored.
         */
        fun hashSignal(signal: FloatArray): String {
            val md = MessageDigest.getInstance("SHA-256")
            for (value in signal) {
                val bits = java.lang.Float.floatToIntBits(value)
                md.update((bits shr 24).toByte())
                md.update((bits shr 16).toByte())
                md.update((bits shr 8).toByte())
                md.update(bits.toByte())
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        /**
         * Generate an ISO 8601 UTC timestamp.
         */
        fun isoTimestamp(): String {
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
            sdf.timeZone = TimeZone.getTimeZone("UTC")
            return sdf.format(Date())
        }
    }
}
