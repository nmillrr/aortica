package io.aortica.ecg.inference

import android.content.Context
import org.json.JSONObject

/**
 * Maps multi-task inference results to simplified risk tiers
 * suitable for community health workers and field clinicians.
 *
 * Reuses the same tier logic as the Python `simplified_output` module (US-060).
 *
 * Tiers:
 * - **LOW** ("Low risk") — no immediate action required
 * - **REFER** ("Refer for assessment") — schedule a clinician follow-up
 * - **URGENT** ("Urgent referral recommended") — seek immediate medical care
 */
object TierMapper {

    /** Risk tier levels ordered by severity. */
    enum class Tier(val key: String) {
        LOW("low"),
        REFER("refer"),
        URGENT("urgent");

        companion object {
            fun fromKey(key: String): Tier = entries.first { it.key == key }
        }
    }

    /**
     * Simplified output report for display.
     *
     * @property tier The assigned risk tier.
     * @property tierLabel Localized tier label (e.g., "Low risk").
     * @property summary 1–2 sentence finding summary.
     * @property actions List of recommended actions.
     * @property keyFindings Top findings driving the tier assignment.
     */
    data class TierReport(
        val tier: Tier,
        val tierLabel: String,
        val summary: String,
        val actions: List<String>,
        val keyFindings: List<KeyFinding>
    )

    /**
     * A single finding contributing to the tier assignment.
     */
    data class KeyFinding(
        val condition: String,
        val confidence: Float,
        val task: String,
        val severity: Tier
    )

    // --- Condition thresholds ---
    // Conditions above these confidence thresholds trigger tier escalation.

    /** Conditions triggering URGENT tier. */
    private val URGENT_CONDITIONS = mapOf(
        // Rhythm
        "VT" to 0.7f,
        "VF" to 0.5f,
        "av_block_3rd" to 0.7f,
        // Ischaemia
        "STEMI" to 0.6f,
        "posterior_MI" to 0.7f,
        "occlusive_NSTEMI" to 0.7f,
        // Risk
        "mortality_1y" to 0.8f
    )

    /** Conditions triggering REFER tier. */
    private val REFER_CONDITIONS = mapOf(
        // Rhythm
        "AF" to 0.6f,
        "AFL" to 0.6f,
        "SVT" to 0.6f,
        "AVNRT" to 0.6f,
        "AVRT" to 0.6f,
        "av_block_2nd" to 0.6f,
        "LBBB" to 0.6f,
        "WPW" to 0.6f,
        "brugada_pattern" to 0.5f,
        "sinus_brady" to 0.7f,
        "sinus_tachy" to 0.7f,
        // Structural
        "LVH" to 0.6f,
        "RVH" to 0.6f,
        "LVSD" to 0.6f,
        "HCM" to 0.6f,
        "ARVC" to 0.6f,
        "amyloidosis" to 0.5f,
        "aortic_stenosis" to 0.6f,
        // Ischaemia
        "old_MI" to 0.6f,
        "hyperkalaemia" to 0.6f,
        "QTc_prolongation" to 0.6f,
        // Risk
        "hf_hosp_12m" to 0.7f,
        "af_onset_12m" to 0.7f
    )

    /** Rhythm class names (must match model output order). */
    private val RHYTHM_CLASSES = listOf(
        "AF", "AFL", "SVT", "AVNRT", "AVRT", "VT", "VF", "idioventricular",
        "sinus_brady", "sinus_tachy", "PAC", "PVC", "av_block_1st", "av_block_2nd",
        "av_block_3rd", "LBBB", "RBBB", "LAFB", "LPFB", "WPW",
        "pacemaker_rhythm", "normal_sinus_rhythm",
        "brugada_pattern", "short_QT_syndrome", "CPVT", "fascicular_VT",
        "atypical_atrial_flutter", "inappropriate_sinus_tachy"
    )

    /** Structural class names. */
    private val STRUCTURAL_CLASSES = listOf(
        "LVH", "RVH", "LVSD", "HFpEF_risk", "DCM", "HCM", "ARVC", "amyloidosis",
        "aortic_stenosis", "mitral_regurgitation", "pulmonary_HTN",
        "LA_enlargement", "RA_enlargement", "pericarditis", "myocarditis",
        "LV_strain_grade", "RV_strain_PE", "Takotsubo_pattern",
        "infiltrative_cardiomyopathy_strain"
    )

    /** Ischaemia class names. */
    private val ISCHAEMIA_CLASSES = listOf(
        "STEMI", "posterior_MI", "occlusive_NSTEMI", "old_MI", "hyperkalaemia",
        "hypokalaemia", "hypercalcaemia", "hypothyroidism_pattern",
        "digitalis_effect", "QTc_prolongation",
        "early_repol_vs_STEMI", "de_Winter_T_wave", "Wellens_syndrome",
        "aVR_ST_elevation", "Sgarbossa_criteria",
        "hyperkalaemia_severity_grade", "hypothermia_osborn_waves",
        "TCA_toxicity", "digoxin_effect_vs_toxicity"
    )

    /** Risk output names. */
    private val RISK_OUTPUTS = listOf(
        "mortality_1y", "hf_hosp_12m", "af_onset_12m",
        "subclinical_lvsd", "conduction_disease_progression", "sudden_cardiac_death"
    )

    /**
     * Map an [InferenceResult] to a [TierReport].
     *
     * @param result Raw inference output.
     * @param context Optional Android context for loading locale strings.
     * @param locale Locale code (default: "en").
     * @return Tier report with severity, summary, and actions.
     */
    fun mapToTier(
        result: InferenceResult,
        context: Context? = null,
        locale: String = "en"
    ): TierReport {
        val findings = mutableListOf<KeyFinding>()

        // Scan all task outputs against thresholds
        collectFindings(result.rhythm, RHYTHM_CLASSES, "rhythm", findings)
        collectFindings(result.structural, STRUCTURAL_CLASSES, "structural", findings)
        collectFindings(result.ischaemia, ISCHAEMIA_CLASSES, "ischaemia", findings)
        collectFindings(result.risk, RISK_OUTPUTS, "risk", findings)

        // Determine overall tier (highest severity finding wins)
        val tier = when {
            findings.any { it.severity == Tier.URGENT } -> Tier.URGENT
            findings.any { it.severity == Tier.REFER } -> Tier.REFER
            else -> Tier.LOW
        }

        // Sort findings by confidence descending, take top 3
        val topFindings = findings
            .sortedByDescending { it.confidence }
            .take(3)

        // Generate localized output
        val localeStrings = loadLocaleStrings(context, locale)
        val tierLabel = localeStrings.optString("tier_${tier.key}", defaultTierLabel(tier))
        val summary = generateSummary(tier, topFindings, localeStrings)
        val actions = generateActions(tier, localeStrings)

        return TierReport(
            tier = tier,
            tierLabel = tierLabel,
            summary = summary,
            actions = actions,
            keyFindings = topFindings
        )
    }

    private fun collectFindings(
        scores: FloatArray,
        classNames: List<String>,
        task: String,
        findings: MutableList<KeyFinding>
    ) {
        for ((i, name) in classNames.withIndex()) {
            if (i >= scores.size) break
            val score = scores[i]

            // Check urgent thresholds first
            val urgentThreshold = URGENT_CONDITIONS[name]
            if (urgentThreshold != null && score >= urgentThreshold) {
                findings.add(KeyFinding(name, score, task, Tier.URGENT))
                continue
            }

            // Check refer thresholds
            val referThreshold = REFER_CONDITIONS[name]
            if (referThreshold != null && score >= referThreshold) {
                findings.add(KeyFinding(name, score, task, Tier.REFER))
            }
        }
    }

    private fun loadLocaleStrings(context: Context?, locale: String): JSONObject {
        if (context == null) return JSONObject()
        return try {
            val fileName = "locales/$locale.json"
            val json = context.assets.open(fileName).bufferedReader().readText()
            JSONObject(json)
        } catch (_: Exception) {
            JSONObject()
        }
    }

    private fun defaultTierLabel(tier: Tier): String = when (tier) {
        Tier.LOW -> "Low risk"
        Tier.REFER -> "Refer for assessment"
        Tier.URGENT -> "Urgent referral recommended"
    }

    private fun generateSummary(
        tier: Tier,
        topFindings: List<KeyFinding>,
        localeStrings: JSONObject
    ): String {
        if (topFindings.isEmpty()) {
            return localeStrings.optString(
                "summary_low_no_findings",
                "No significant abnormalities detected. Standard ECG findings within normal limits."
            )
        }

        val findingNames = topFindings.joinToString(", ") {
            it.condition.replace("_", " ")
        }

        return when (tier) {
            Tier.URGENT -> localeStrings.optString(
                "summary_urgent",
                "Urgent findings detected: $findingNames. Immediate medical evaluation recommended."
            ).replace("{findings}", findingNames)

            Tier.REFER -> localeStrings.optString(
                "summary_refer",
                "Abnormalities detected: $findingNames. Follow-up with a clinician is recommended."
            ).replace("{findings}", findingNames)

            Tier.LOW -> localeStrings.optString(
                "summary_low",
                "Minor findings noted: $findingNames. No immediate action required."
            ).replace("{findings}", findingNames)
        }
    }

    private fun generateActions(
        tier: Tier,
        localeStrings: JSONObject
    ): List<String> = when (tier) {
        Tier.URGENT -> listOf(
            localeStrings.optString("action_urgent_1", "Seek immediate medical care"),
            localeStrings.optString("action_urgent_2", "Do not leave the patient unattended"),
            localeStrings.optString("action_urgent_3", "Prepare for potential emergency transport")
        )
        Tier.REFER -> listOf(
            localeStrings.optString("action_refer_1", "Schedule a follow-up with a clinician"),
            localeStrings.optString("action_refer_2", "Repeat ECG if symptoms change"),
            localeStrings.optString("action_refer_3", "Document symptoms and findings")
        )
        Tier.LOW -> listOf(
            localeStrings.optString("action_low_1", "No immediate action required"),
            localeStrings.optString("action_low_2", "Continue routine monitoring")
        )
    }
}
