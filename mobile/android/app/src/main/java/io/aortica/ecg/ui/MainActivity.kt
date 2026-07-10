package io.aortica.ecg.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.lifecycle.lifecycleScope
import io.aortica.ecg.AorticaApplication
import io.aortica.ecg.R
import io.aortica.ecg.inference.EcgFileReader
import io.aortica.ecg.inference.SignalPreprocessor
import io.aortica.ecg.inference.TierMapper
import kotlinx.coroutines.launch

/**
 * Main activity for the Aortica ECG mobile app.
 *
 * Provides:
 * - ECG file import (CSV, WFDB via file picker)
 * - AI inference with plain-language tier output
 * - High-contrast, large-touch-target UI for field use
 */
class MainActivity : AppCompatActivity() {

    // UI elements
    private lateinit var btnImportFile: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var cardResult: CardView
    private lateinit var tvTierLabel: TextView
    private lateinit var tvSummary: TextView
    private lateinit var tvActions: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvFindings: TextView
    private lateinit var tvStatus: TextView

    private val app: AorticaApplication
        get() = application as AorticaApplication

    // File picker launcher
    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { processEcgFile(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind views
        btnImportFile = findViewById(R.id.btn_import_file)
        progressBar = findViewById(R.id.progress_bar)
        cardResult = findViewById(R.id.card_result)
        tvTierLabel = findViewById(R.id.tv_tier_label)
        tvSummary = findViewById(R.id.tv_summary)
        tvActions = findViewById(R.id.tv_actions)
        tvLatency = findViewById(R.id.tv_latency)
        tvFindings = findViewById(R.id.tv_findings)
        tvStatus = findViewById(R.id.tv_status)

        // Import button
        btnImportFile.setOnClickListener {
            filePickerLauncher.launch("*/*")
        }

        // Pre-load model in background
        lifecycleScope.launch {
            try {
                tvStatus.text = getString(R.string.status_loading_model)
                app.inferenceEngine.loadModel()
                tvStatus.text = getString(R.string.status_ready)
            } catch (e: Exception) {
                tvStatus.text = getString(R.string.status_model_error, e.message ?: "Unknown")
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                startActivity(Intent(this, SettingsActivity::class.java))
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun processEcgFile(uri: Uri) {
        lifecycleScope.launch {
            showLoading(true)
            cardResult.visibility = View.GONE

            try {
                // Read ECG file
                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw IllegalStateException("Cannot open file")

                val fileName = uri.lastPathSegment ?: "unknown"
                val ecgData = when {
                    fileName.endsWith(".csv", ignoreCase = true) ->
                        EcgFileReader.readCsv(inputStream)
                    fileName.endsWith(".dat", ignoreCase = true) ->
                        EcgFileReader.readWfdbDat(inputStream)
                    else ->
                        // Try CSV as default
                        EcgFileReader.readCsv(inputStream)
                }

                if (ecgData.numSamples == 0) {
                    tvStatus.text = getString(R.string.error_empty_file)
                    showLoading(false)
                    return@launch
                }

                // Preprocess
                val preprocessed = SignalPreprocessor.preprocess(
                    signal = ecgData.signal,
                    numLeads = ecgData.numLeads,
                    numSamples = ecgData.numSamples,
                    sourceSampleRate = ecgData.sampleRate
                )

                // Run inference
                val result = app.inferenceEngine.predict(
                    signal = preprocessed,
                    numLeads = ecgData.numLeads,
                    numSamples = SignalPreprocessor.TARGET_SAMPLES
                )

                // Map to tier
                val tierReport = TierMapper.mapToTier(
                    result = result,
                    context = this@MainActivity,
                    locale = getPreferredLocale()
                )

                // Log to audit
                app.auditLogger.logInference(
                    inputSignal = preprocessed,
                    tier = tierReport.tier,
                    latencyMs = result.latencyMs,
                    numLeads = ecgData.numLeads,
                    numSamples = ecgData.numSamples
                )

                // Display results
                displayResult(tierReport, result.latencyMs)

            } catch (e: Exception) {
                tvStatus.text = getString(R.string.error_inference, e.message ?: "Unknown error")
            } finally {
                showLoading(false)
            }
        }
    }

    private fun displayResult(report: TierMapper.TierReport, latencyMs: Double) {
        cardResult.visibility = View.VISIBLE

        // Set tier label with color
        tvTierLabel.text = report.tierLabel
        val tierColor = when (report.tier) {
            TierMapper.Tier.LOW -> getColor(R.color.tier_low)
            TierMapper.Tier.REFER -> getColor(R.color.tier_refer)
            TierMapper.Tier.URGENT -> getColor(R.color.tier_urgent)
        }
        tvTierLabel.setTextColor(tierColor)
        cardResult.setCardBackgroundColor(
            when (report.tier) {
                TierMapper.Tier.LOW -> getColor(R.color.tier_low_bg)
                TierMapper.Tier.REFER -> getColor(R.color.tier_refer_bg)
                TierMapper.Tier.URGENT -> getColor(R.color.tier_urgent_bg)
            }
        )

        tvSummary.text = report.summary
        tvActions.text = report.actions.joinToString("\n") { "• $it" }

        if (report.keyFindings.isNotEmpty()) {
            tvFindings.visibility = View.VISIBLE
            tvFindings.text = report.keyFindings.joinToString("\n") {
                "• ${it.condition.replace("_", " ")}: ${(it.confidence * 100).toInt()}%"
            }
        } else {
            tvFindings.visibility = View.GONE
        }

        tvLatency.text = getString(R.string.latency_label, latencyMs)
        tvStatus.text = getString(R.string.status_complete)
    }

    private fun showLoading(loading: Boolean) {
        progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        btnImportFile.isEnabled = !loading
        if (loading) {
            tvStatus.text = getString(R.string.status_analyzing)
        }
    }

    private fun getPreferredLocale(): String {
        val prefs = getSharedPreferences("aortica_prefs", MODE_PRIVATE)
        return prefs.getString("language", "en") ?: "en"
    }
}
