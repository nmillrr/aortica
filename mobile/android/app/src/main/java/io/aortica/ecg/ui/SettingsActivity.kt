package io.aortica.ecg.ui

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.aortica.ecg.AorticaApplication
import io.aortica.ecg.R
import io.aortica.ecg.audit.AuditSyncService
import kotlinx.coroutines.launch

/**
 * Settings screen for configuring:
 * - Remote sync URL
 * - Device ID
 * - Sync frequency
 * - Language selection
 * - High-contrast mode
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences
    private lateinit var etSyncUrl: EditText
    private lateinit var etDeviceId: EditText
    private lateinit var spinnerSyncFreq: Spinner
    private lateinit var spinnerLanguage: Spinner
    private lateinit var switchHighContrast: Switch
    private lateinit var btnSave: Button
    private lateinit var btnSyncNow: Button
    private lateinit var tvSyncStatus: TextView
    private lateinit var tvAuditCount: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

        // Bind views
        etSyncUrl = findViewById(R.id.et_sync_url)
        etDeviceId = findViewById(R.id.et_device_id)
        spinnerSyncFreq = findViewById(R.id.spinner_sync_freq)
        spinnerLanguage = findViewById(R.id.spinner_language)
        switchHighContrast = findViewById(R.id.switch_high_contrast)
        btnSave = findViewById(R.id.btn_save)
        btnSyncNow = findViewById(R.id.btn_sync_now)
        tvSyncStatus = findViewById(R.id.tv_sync_status)
        tvAuditCount = findViewById(R.id.tv_audit_count)

        // Setup spinners
        setupSyncFrequencySpinner()
        setupLanguageSpinner()

        // Load saved preferences
        loadPreferences()

        // Buttons
        btnSave.setOnClickListener { savePreferences() }
        btnSyncNow.setOnClickListener { syncNow() }

        // Show audit stats
        updateAuditStats()
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }

    private fun setupSyncFrequencySpinner() {
        val frequencies = arrayOf(
            "Every 15 minutes",
            "Every hour",
            "Every 6 hours",
            "Daily",
            "Manual only"
        )
        spinnerSyncFreq.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            frequencies
        )
    }

    private fun setupLanguageSpinner() {
        val languages = arrayOf(
            "English",
            "Français",
            "Español",
            "Kiswahili"
        )
        spinnerLanguage.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            languages
        )
    }

    private fun loadPreferences() {
        etSyncUrl.setText(prefs.getString(KEY_SYNC_URL, DEFAULT_SYNC_URL))
        etDeviceId.setText(prefs.getString(KEY_DEVICE_ID, ""))
        spinnerSyncFreq.setSelection(prefs.getInt(KEY_SYNC_FREQ, 2))
        switchHighContrast.isChecked = prefs.getBoolean(KEY_HIGH_CONTRAST, false)

        val locale = prefs.getString(KEY_LANGUAGE, "en") ?: "en"
        spinnerLanguage.setSelection(
            when (locale) {
                "fr" -> 1
                "es" -> 2
                "sw" -> 3
                else -> 0
            }
        )
    }

    private fun savePreferences() {
        prefs.edit().apply {
            putString(KEY_SYNC_URL, etSyncUrl.text.toString().trim())
            putString(KEY_DEVICE_ID, etDeviceId.text.toString().trim())
            putInt(KEY_SYNC_FREQ, spinnerSyncFreq.selectedItemPosition)
            putBoolean(KEY_HIGH_CONTRAST, switchHighContrast.isChecked)
            putString(KEY_LANGUAGE, when (spinnerLanguage.selectedItemPosition) {
                1 -> "fr"
                2 -> "es"
                3 -> "sw"
                else -> "en"
            })
            apply()
        }

        Toast.makeText(this, R.string.settings_saved, Toast.LENGTH_SHORT).show()
    }

    private fun syncNow() {
        val syncUrl = etSyncUrl.text.toString().trim()
        val deviceId = etDeviceId.text.toString().trim()

        if (syncUrl.isEmpty()) {
            tvSyncStatus.text = getString(R.string.sync_no_url)
            return
        }

        lifecycleScope.launch {
            tvSyncStatus.text = getString(R.string.sync_in_progress)
            btnSyncNow.isEnabled = false

            val app = application as AorticaApplication
            val syncService = AuditSyncService(this@SettingsActivity, app.auditLogger)
            val result = syncService.sync(syncUrl, deviceId)

            tvSyncStatus.text = if (result.success) {
                getString(R.string.sync_success, result.syncedCount, result.pendingCount)
            } else {
                getString(R.string.sync_error, result.error ?: "Unknown")
            }

            btnSyncNow.isEnabled = true
            updateAuditStats()
        }
    }

    private fun updateAuditStats() {
        val app = application as AorticaApplication
        val total = app.auditLogger.totalCount()
        val pending = app.auditLogger.pendingCount()
        tvAuditCount.text = getString(R.string.audit_stats, total, pending)
    }

    companion object {
        const val PREFS_NAME = "aortica_prefs"
        const val KEY_SYNC_URL = "sync_url"
        const val KEY_DEVICE_ID = "device_id"
        const val KEY_SYNC_FREQ = "sync_frequency"
        const val KEY_LANGUAGE = "language"
        const val KEY_HIGH_CONTRAST = "high_contrast"
        const val DEFAULT_SYNC_URL = ""
    }
}
