package io.aortica.ecg.update

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for [ModelUpdateManager] version derivation,
 * [ModelManifest] JSON parsing, and OTA update logic.
 *
 * These run on the JVM (no Android instrumentation needed).
 */
class ModelUpdateManagerTest {

    // -----------------------------------------------------------------------
    // Version derivation tests
    // -----------------------------------------------------------------------

    @Test
    fun `deriveVersionCode from standard tag v0_3_0`() {
        assertEquals(300, ModelUpdateManager.deriveVersionCode("v0.3.0"))
    }

    @Test
    fun `deriveVersionCode from tag v2_0_0`() {
        assertEquals(20000, ModelUpdateManager.deriveVersionCode("v2.0.0"))
    }

    @Test
    fun `deriveVersionCode from tag v1_2_3`() {
        assertEquals(10203, ModelUpdateManager.deriveVersionCode("v1.2.3"))
    }

    @Test
    fun `deriveVersionCode without v prefix`() {
        assertEquals(10203, ModelUpdateManager.deriveVersionCode("1.2.3"))
    }

    @Test
    fun `deriveVersionCode from tag v0_0_1`() {
        assertEquals(1, ModelUpdateManager.deriveVersionCode("v0.0.1"))
    }

    @Test
    fun `deriveVersionCode from major only`() {
        assertEquals(30000, ModelUpdateManager.deriveVersionCode("v3"))
    }

    @Test
    fun `deriveVersionCode from major_minor only`() {
        assertEquals(30200, ModelUpdateManager.deriveVersionCode("v3.2"))
    }

    @Test
    fun `deriveVersionCode fallback for invalid tag`() {
        assertEquals(100, ModelUpdateManager.deriveVersionCode("invalid"))
    }

    @Test
    fun `deriveVersionCode fallback for empty string`() {
        assertEquals(100, ModelUpdateManager.deriveVersionCode(""))
    }

    @Test
    fun `deriveVersionName strips v prefix`() {
        assertEquals("0.3.0", ModelUpdateManager.deriveVersionName("v0.3.0"))
    }

    @Test
    fun `deriveVersionName preserves name without v prefix`() {
        assertEquals("1.2.3", ModelUpdateManager.deriveVersionName("1.2.3"))
    }

    // -----------------------------------------------------------------------
    // Version comparison tests
    // -----------------------------------------------------------------------

    @Test
    fun `compareVersions equal versions`() {
        assertEquals(0, ModelUpdateManager.compareVersions("1.2.3", "1.2.3"))
    }

    @Test
    fun `compareVersions newer major`() {
        assertTrue(ModelUpdateManager.compareVersions("2.0.0", "1.9.9") > 0)
    }

    @Test
    fun `compareVersions newer minor`() {
        assertTrue(ModelUpdateManager.compareVersions("1.3.0", "1.2.9") > 0)
    }

    @Test
    fun `compareVersions newer patch`() {
        assertTrue(ModelUpdateManager.compareVersions("1.2.4", "1.2.3") > 0)
    }

    @Test
    fun `compareVersions older version`() {
        assertTrue(ModelUpdateManager.compareVersions("0.1.0", "0.2.0") < 0)
    }

    @Test
    fun `compareVersions different length`() {
        assertEquals(0, ModelUpdateManager.compareVersions("1.0", "1.0.0"))
    }

    @Test
    fun `isNewerVersion returns true for newer`() {
        assertTrue(ModelUpdateManager.isNewerVersion("0.3.0", "0.2.0"))
    }

    @Test
    fun `isNewerVersion returns false for older`() {
        assertFalse(ModelUpdateManager.isNewerVersion("0.1.0", "0.2.0"))
    }

    @Test
    fun `isNewerVersion returns false for same`() {
        assertFalse(ModelUpdateManager.isNewerVersion("0.2.0", "0.2.0"))
    }

    @Test
    fun `isNewerVersion returns true when current is bundled`() {
        assertTrue(ModelUpdateManager.isNewerVersion("0.1.0", "bundled"))
    }

    // -----------------------------------------------------------------------
    // SHA-256 computation tests
    // -----------------------------------------------------------------------

    @Test
    fun `computeSha256 produces correct hash`() {
        val tmpFile = java.io.File.createTempFile("test_sha256", ".bin")
        try {
            tmpFile.writeBytes("hello world".toByteArray())
            val hash = ModelUpdateManager.computeSha256(tmpFile)
            // SHA-256 of "hello world"
            assertEquals(
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
                hash
            )
        } finally {
            tmpFile.delete()
        }
    }

    @Test
    fun `computeSha256 produces different hash for different content`() {
        val file1 = java.io.File.createTempFile("test1", ".bin")
        val file2 = java.io.File.createTempFile("test2", ".bin")
        try {
            file1.writeBytes("content A".toByteArray())
            file2.writeBytes("content B".toByteArray())
            assertNotEquals(
                ModelUpdateManager.computeSha256(file1),
                ModelUpdateManager.computeSha256(file2)
            )
        } finally {
            file1.delete()
            file2.delete()
        }
    }

    // -----------------------------------------------------------------------
    // ModelManifest JSON parsing tests
    // -----------------------------------------------------------------------

    @Test
    fun `ModelManifest fromJson parses valid JSON`() {
        val json = """
        {
            "latest_version": "0.3.0",
            "download_url": "https://example.com/model.onnx",
            "sha256": "abc123def456",
            "min_app_version": "1.0.0",
            "file_size_bytes": 15000000
        }
        """.trimIndent()

        val manifest = ModelManifest.fromJson(json)

        assertEquals("0.3.0", manifest.latestVersion)
        assertEquals("https://example.com/model.onnx", manifest.downloadUrl)
        assertEquals("abc123def456", manifest.sha256)
        assertEquals("1.0.0", manifest.minAppVersion)
        assertEquals(15000000L, manifest.fileSizeBytes)
    }

    @Test
    fun `ModelManifest fromJson handles missing optional fields`() {
        val json = """
        {
            "latest_version": "0.2.0",
            "download_url": "https://example.com/model.onnx",
            "sha256": "deadbeef",
            "min_app_version": "1.0.0"
        }
        """.trimIndent()

        val manifest = ModelManifest.fromJson(json)

        assertEquals("0.2.0", manifest.latestVersion)
        assertEquals(0L, manifest.fileSizeBytes)
    }

    @Test(expected = org.json.JSONException::class)
    fun `ModelManifest fromJson throws on missing required field`() {
        val json = """
        {
            "latest_version": "0.2.0",
            "sha256": "deadbeef",
            "min_app_version": "1.0.0"
        }
        """.trimIndent()

        // Missing download_url should throw
        ModelManifest.fromJson(json)
    }

    @Test
    fun `ModelManifest roundtrip toJson and fromJson`() {
        val original = ModelManifest(
            latestVersion = "0.4.0",
            downloadUrl = "https://hf.co/model.onnx",
            sha256 = "sha256hash",
            minAppVersion = "1.1.0",
            fileSizeBytes = 25000000
        )

        val json = original.toJson()
        val parsed = ModelManifest.fromJson(json)

        assertEquals(original.latestVersion, parsed.latestVersion)
        assertEquals(original.downloadUrl, parsed.downloadUrl)
        assertEquals(original.sha256, parsed.sha256)
        assertEquals(original.minAppVersion, parsed.minAppVersion)
        assertEquals(original.fileSizeBytes, parsed.fileSizeBytes)
    }

    @Test
    fun `ModelManifest toJson produces valid JSON`() {
        val manifest = ModelManifest(
            latestVersion = "0.3.0",
            downloadUrl = "https://example.com/model.onnx",
            sha256 = "abc123",
            minAppVersion = "1.0.0",
            fileSizeBytes = 10000
        )

        val json = manifest.toJson()

        // Should be parseable
        val obj = org.json.JSONObject(json)
        assertEquals("0.3.0", obj.getString("latest_version"))
        assertEquals("https://example.com/model.onnx", obj.getString("download_url"))
        assertEquals("abc123", obj.getString("sha256"))
    }
}
