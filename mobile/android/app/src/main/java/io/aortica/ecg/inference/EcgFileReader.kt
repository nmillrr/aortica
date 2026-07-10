package io.aortica.ecg.inference

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Reads ECG signals from CSV and WFDB-format files on Android.
 *
 * Supports:
 * - CSV files with one column per lead (header row optional)
 * - Simple WFDB .dat files (16-bit signed integer, little-endian)
 *
 * For more complex formats, the server-side Python pipeline should be used.
 */
object EcgFileReader {

    /**
     * Parsed ECG data from a file.
     *
     * @property signal Flat array of shape [numLeads * numSamples]
     * @property numLeads Number of leads detected
     * @property numSamples Number of samples per lead
     * @property sampleRate Sample rate in Hz (from header or default 500)
     */
    data class EcgData(
        val signal: FloatArray,
        val numLeads: Int,
        val numSamples: Int,
        val sampleRate: Int = 500
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is EcgData) return false
            return signal.contentEquals(other.signal) &&
                numLeads == other.numLeads &&
                numSamples == other.numSamples &&
                sampleRate == other.sampleRate
        }

        override fun hashCode(): Int {
            var result = signal.contentHashCode()
            result = 31 * result + numLeads
            result = 31 * result + numSamples
            result = 31 * result + sampleRate
            return result
        }
    }

    /**
     * Read ECG data from a CSV input stream.
     *
     * Expected format: one column per lead, one row per sample.
     * The first row is treated as a header if it contains non-numeric values.
     *
     * @param inputStream CSV data stream
     * @param sampleRate Sample rate in Hz (default 500)
     * @param delimiter Column delimiter (default ',')
     * @return Parsed [EcgData]
     */
    fun readCsv(
        inputStream: InputStream,
        sampleRate: Int = 500,
        delimiter: Char = ','
    ): EcgData {
        val reader = BufferedReader(InputStreamReader(inputStream))
        val rows = mutableListOf<FloatArray>()
        var numLeads = 0

        reader.useLines { lines ->
            for (line in lines) {
                val trimmed = line.trim()
                if (trimmed.isEmpty()) continue

                val parts = trimmed.split(delimiter).map { it.trim() }

                // Skip header row (contains non-numeric values)
                if (rows.isEmpty()) {
                    val isHeader = parts.any { part ->
                        part.toFloatOrNull() == null && part.isNotEmpty()
                    }
                    if (isHeader) {
                        numLeads = parts.size
                        continue
                    }
                }

                val values = parts.mapNotNull { it.toFloatOrNull() }.toFloatArray()
                if (values.isNotEmpty()) {
                    if (numLeads == 0) numLeads = values.size
                    rows.add(values)
                }
            }
        }

        if (rows.isEmpty()) {
            return EcgData(FloatArray(0), 0, 0, sampleRate)
        }

        val numSamples = rows.size

        // Transpose from [samples, leads] to [leads, samples] (flat)
        val signal = FloatArray(numLeads * numSamples)
        for (s in 0 until numSamples) {
            for (l in 0 until numLeads) {
                val value = if (l < rows[s].size) rows[s][l] else 0f
                signal[l * numSamples + s] = value
            }
        }

        return EcgData(signal, numLeads, numSamples, sampleRate)
    }

    /**
     * Read ECG data from a raw binary WFDB .dat file.
     *
     * Assumes 16-bit signed integers in little-endian byte order (format 16).
     * This is the most common WFDB storage format.
     *
     * @param inputStream Binary data stream
     * @param numLeads Number of leads (channels) in the file
     * @param sampleRate Sample rate in Hz
     * @param gain ADC gain (units per mV, default 200 for standard WFDB)
     * @return Parsed [EcgData] with signal in mV
     */
    fun readWfdbDat(
        inputStream: InputStream,
        numLeads: Int = 1,
        sampleRate: Int = 500,
        gain: Float = 200f
    ): EcgData {
        val bytes = inputStream.readBytes()
        val totalSamples = bytes.size / (2 * numLeads)

        if (totalSamples == 0) {
            return EcgData(FloatArray(0), numLeads, 0, sampleRate)
        }

        // Read interleaved 16-bit samples → [leads, samples] flat layout
        val signal = FloatArray(numLeads * totalSamples)

        for (s in 0 until totalSamples) {
            for (l in 0 until numLeads) {
                val offset = (s * numLeads + l) * 2
                if (offset + 1 < bytes.size) {
                    val low = bytes[offset].toInt() and 0xFF
                    val high = bytes[offset + 1].toInt()
                    val rawValue = (high shl 8) or low  // Little-endian signed 16-bit
                    signal[l * totalSamples + s] = rawValue.toShort().toFloat() / gain
                }
            }
        }

        return EcgData(signal, numLeads, totalSamples, sampleRate)
    }
}
