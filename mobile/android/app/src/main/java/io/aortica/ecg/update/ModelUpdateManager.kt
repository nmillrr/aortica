package io.aortica.ecg.update

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Data class representing the model manifest returned by the server.
 *
 * The server endpoint `POST /api/v1/mobile/model-manifest` returns this
 * information so the app can decide whether an OTA model update is needed.
 *
 * @property latestVersion Semantic version of the latest model (e.g. "0.3.0").
 * @property downloadUrl URL to download the ONNX model file.
 * @property sha256 SHA-256 hex digest of the model file for integrity check.
 * @property minAppVersion Minimum app version required for this model.
 * @property fileSizeBytes Size of the model file in bytes.
 */
data class ModelManifest(
    val latestVersion: String,
    val downloadUrl: String,
    val sha256: String,
    val minAppVersion: String,
    val fileSizeBytes: Long = 0
) {
    companion object {
        /**
         * Parse a [ModelManifest] from a JSON string.
         *
         * @param json Raw JSON string from the server response.
         * @return Parsed [ModelManifest].
         * @throws org.json.JSONException if required fields are missing.
         */
        fun fromJson(json: String): ModelManifest {
            val obj = JSONObject(json)
            return ModelManifest(
                latestVersion = obj.getString("latest_version"),
                downloadUrl = obj.getString("download_url"),
                sha256 = obj.getString("sha256"),
                minAppVersion = obj.getString("min_app_version"),
                fileSizeBytes = obj.optLong("file_size_bytes", 0)
            )
        }
    }

    /**
     * Serialize this manifest to a JSON string.
     */
    fun toJson(): String {
        val obj = JSONObject()
        obj.put("latest_version", latestVersion)
        obj.put("download_url", downloadUrl)
        obj.put("sha256", sha256)
        obj.put("min_app_version", minAppVersion)
        obj.put("file_size_bytes", fileSizeBytes)
        return obj.toString()
    }
}

/**
 * Manages OTA (over-the-air) model updates for the Aortica Android app.
 *
 * On startup (when online), the app checks the configured server endpoint
 * for a newer model version. If available, it downloads and caches the new
 * ONNX model locally. The inference engine can then load the cached model
 * instead of the bundled one.
 *
 * Falls back to the bundled model in assets/ if:
 * - No network connectivity
 * - Download fails
 * - Integrity check (SHA-256) fails
 * - Server is unreachable
 *
 * @param context Application context.
 * @param manifestUrl Full URL of the model manifest endpoint.
 * @param modelCacheDir Directory to cache downloaded models.
 */
class ModelUpdateManager(
    private val context: Context,
    private val manifestUrl: String,
    private val modelCacheDir: File = File(context.filesDir, "models")
) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /**
     * Check the server for a newer model version.
     *
     * @return [ModelManifest] if a newer version is available, null otherwise.
     */
    suspend fun checkForUpdate(): ModelManifest? = withContext(Dispatchers.IO) {
        try {
            val manifest = fetchManifest() ?: return@withContext null
            val currentVersion = getCurrentModelVersion()

            if (isNewerVersion(manifest.latestVersion, currentVersion) &&
                isAppVersionCompatible(manifest.minAppVersion)
            ) {
                manifest
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Download and cache a model update.
     *
     * @param manifest The manifest describing the model to download.
     * @return Path to the downloaded model file, or null if download/verification failed.
     */
    suspend fun downloadUpdate(manifest: ModelManifest): File? = withContext(Dispatchers.IO) {
        try {
            modelCacheDir.mkdirs()
            val tempFile = File(modelCacheDir, "model_${manifest.latestVersion}.onnx.tmp")
            val finalFile = File(modelCacheDir, "model_${manifest.latestVersion}.onnx")

            // Skip if already downloaded and verified
            if (finalFile.exists() && computeSha256(finalFile) == manifest.sha256) {
                setCurrentModelVersion(manifest.latestVersion)
                setCachedModelPath(finalFile.absolutePath)
                return@withContext finalFile
            }

            // Download to temp file
            val url = URL(manifest.downloadUrl)
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = CONNECT_TIMEOUT_MS
            connection.readTimeout = READ_TIMEOUT_MS

            try {
                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        input.copyTo(output, bufferSize = 8192)
                    }
                }
            } finally {
                connection.disconnect()
            }

            // Verify SHA-256
            val actualSha = computeSha256(tempFile)
            if (actualSha != manifest.sha256) {
                tempFile.delete()
                return@withContext null
            }

            // Rename temp → final
            tempFile.renameTo(finalFile)

            // Update preferences
            setCurrentModelVersion(manifest.latestVersion)
            setCachedModelPath(finalFile.absolutePath)

            finalFile
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Get the path to the best available model.
     *
     * Returns the cached OTA model path if available, otherwise null
     * (caller should fall back to the bundled assets model).
     */
    fun getCachedModelPath(): String? {
        val path = prefs.getString(KEY_CACHED_MODEL_PATH, null)
        if (path != null && File(path).exists()) {
            return path
        }
        return null
    }

    /**
     * Get the current model version string.
     *
     * Returns "bundled" if no OTA update has been applied.
     */
    fun getCurrentModelVersion(): String {
        return prefs.getString(KEY_CURRENT_VERSION, VERSION_BUNDLED) ?: VERSION_BUNDLED
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    internal fun fetchManifest(): ModelManifest? {
        val url = URL(manifestUrl)
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.connectTimeout = CONNECT_TIMEOUT_MS
        connection.readTimeout = READ_TIMEOUT_MS

        return try {
            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                return null
            }
            val body = connection.inputStream.bufferedReader().use { it.readText() }
            ModelManifest.fromJson(body)
        } catch (e: Exception) {
            null
        } finally {
            connection.disconnect()
        }
    }

    private fun setCurrentModelVersion(version: String) {
        prefs.edit().putString(KEY_CURRENT_VERSION, version).apply()
    }

    private fun setCachedModelPath(path: String) {
        prefs.edit().putString(KEY_CACHED_MODEL_PATH, path).apply()
    }

    private fun isAppVersionCompatible(minVersion: String): Boolean {
        return compareVersions(getAppVersion(), minVersion) >= 0
    }

    private fun getAppVersion(): String {
        return try {
            val info = context.packageManager.getPackageInfo(context.packageName, 0)
            info.versionName ?: VERSION_BUNDLED
        } catch (e: Exception) {
            VERSION_BUNDLED
        }
    }

    companion object {
        private const val PREFS_NAME = "aortica_model_update"
        private const val KEY_CURRENT_VERSION = "current_model_version"
        private const val KEY_CACHED_MODEL_PATH = "cached_model_path"
        private const val VERSION_BUNDLED = "bundled"
        private const val CONNECT_TIMEOUT_MS = 10_000
        private const val READ_TIMEOUT_MS = 60_000

        /**
         * Compare two semantic version strings.
         *
         * @return Positive if v1 > v2, negative if v1 < v2, zero if equal.
         */
        fun compareVersions(v1: String, v2: String): Int {
            val parts1 = v1.split(".").map { it.toIntOrNull() ?: 0 }
            val parts2 = v2.split(".").map { it.toIntOrNull() ?: 0 }
            val maxLen = maxOf(parts1.size, parts2.size)

            for (i in 0 until maxLen) {
                val p1 = parts1.getOrElse(i) { 0 }
                val p2 = parts2.getOrElse(i) { 0 }
                if (p1 != p2) return p1 - p2
            }
            return 0
        }

        /**
         * Check if newVersion is strictly newer than currentVersion.
         *
         * The special value "bundled" is always considered older than
         * any numbered version.
         */
        fun isNewerVersion(newVersion: String, currentVersion: String): Boolean {
            if (currentVersion == VERSION_BUNDLED) return true
            return compareVersions(newVersion, currentVersion) > 0
        }

        /**
         * Compute SHA-256 hex digest of a file.
         */
        fun computeSha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    digest.update(buffer, 0, bytesRead)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        /**
         * Derive versionCode from a git tag string.
         *
         * e.g. "v0.3.0" → 300, "v2.0.0" → 20000, "1.2.3" → 10203
         * Falls back to 100 for unparseable tags.
         */
        fun deriveVersionCode(tag: String): Int {
            val stripped = if (tag.startsWith("v")) tag.substring(1) else tag
            val parts = stripped.split(".")
            return try {
                when {
                    parts.size >= 3 -> parts[0].toInt() * 10000 + parts[1].toInt() * 100 + parts[2].toInt()
                    parts.size == 2 -> parts[0].toInt() * 10000 + parts[1].toInt() * 100
                    parts.size == 1 -> parts[0].toInt() * 10000
                    else -> 100
                }
            } catch (e: NumberFormatException) {
                100
            }
        }

        /**
         * Derive versionName from a git tag string.
         *
         * Strips the leading "v" if present. e.g. "v0.3.0" → "0.3.0"
         */
        fun deriveVersionName(tag: String): String {
            return if (tag.startsWith("v")) tag.substring(1) else tag
        }
    }
}
