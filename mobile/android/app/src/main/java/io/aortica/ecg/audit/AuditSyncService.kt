package io.aortica.ecg.audit

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Handles syncing audit log entries to a remote endpoint.
 *
 * Reuses the sync protocol from US-055/US-056:
 * - POST JSON array of anonymized audit entries
 * - Expects 200 OK with JSON response containing synced IDs
 * - Strips all patient-identifiable metadata before upload
 *
 * @param context Application context for connectivity checking
 * @param auditLogger The audit logger to sync from
 */
class AuditSyncService(
    private val context: Context,
    private val auditLogger: AuditLogger
) {

    /**
     * Result of a sync attempt.
     */
    data class SyncResult(
        val success: Boolean,
        val syncedCount: Int = 0,
        val pendingCount: Int = 0,
        val error: String? = null
    )

    /**
     * Attempt to sync pending audit entries to the configured remote endpoint.
     *
     * Only proceeds if network connectivity is available.
     *
     * @param syncUrl Remote endpoint URL for audit sync
     * @param deviceId Device identifier for the sync payload
     * @param batchSize Maximum entries per sync batch
     * @return [SyncResult] with outcome
     */
    suspend fun sync(
        syncUrl: String,
        deviceId: String,
        batchSize: Int = 50
    ): SyncResult = withContext(Dispatchers.IO) {
        // Check connectivity
        if (!isNetworkAvailable()) {
            return@withContext SyncResult(
                success = false,
                pendingCount = auditLogger.pendingCount(),
                error = "No network connectivity"
            )
        }

        val pending = auditLogger.getPendingSyncBatch(batchSize)
        if (pending.length() == 0) {
            return@withContext SyncResult(
                success = true,
                syncedCount = 0,
                pendingCount = 0
            )
        }

        try {
            // Build sync payload
            val payload = JSONObject().apply {
                put("device_id", deviceId)
                put("entries", pending)
                put("client_type", "android")
                put("client_version", "1.0.0")
            }

            // POST to sync endpoint
            val url = URL(syncUrl)
            val connection = url.openConnection() as HttpURLConnection
            connection.apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Accept", "application/json")
                connectTimeout = 15_000
                readTimeout = 15_000
                doOutput = true
            }

            OutputStreamWriter(connection.outputStream).use { writer ->
                writer.write(payload.toString())
                writer.flush()
            }

            val responseCode = connection.responseCode

            if (responseCode == 200) {
                // Parse response for synced IDs
                val responseBody = connection.inputStream.bufferedReader().readText()
                val response = JSONObject(responseBody)
                val syncedIds = response.optJSONArray("synced_ids")

                if (syncedIds != null) {
                    val ids = (0 until syncedIds.length()).map {
                        syncedIds.getLong(it)
                    }
                    auditLogger.markSynced(ids)

                    return@withContext SyncResult(
                        success = true,
                        syncedCount = ids.size,
                        pendingCount = auditLogger.pendingCount()
                    )
                }

                // If no synced_ids in response, mark all as synced
                val ids = (0 until pending.length()).map {
                    pending.getJSONObject(it).getLong("id")
                }
                auditLogger.markSynced(ids)

                return@withContext SyncResult(
                    success = true,
                    syncedCount = ids.size,
                    pendingCount = auditLogger.pendingCount()
                )
            } else {
                return@withContext SyncResult(
                    success = false,
                    pendingCount = auditLogger.pendingCount(),
                    error = "HTTP $responseCode"
                )
            }
        } catch (e: Exception) {
            return@withContext SyncResult(
                success = false,
                pendingCount = auditLogger.pendingCount(),
                error = e.message ?: "Unknown error"
            )
        }
    }

    /**
     * Check if the device has network connectivity.
     */
    fun isNetworkAvailable(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false

        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false

        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }
}
