package io.aortica.ecg.inference

/**
 * Extensible interface for Bluetooth LE ECG device connectivity.
 *
 * Implementations of this interface connect to specific BLE ECG devices
 * (e.g., AliveCor KardiaMobile) and stream raw signal data for inference.
 *
 * This is a protocol stub — concrete implementations are device-specific
 * and will be added as device partnerships are established.
 */
interface BleEcgDevice {

    /** Human-readable device name. */
    val deviceName: String

    /** BLE service UUID used to discover this device type. */
    val serviceUuid: String

    /** Number of leads this device captures (typically 1 or 6). */
    val leadCount: Int

    /** Native sample rate of the device in Hz. */
    val sampleRate: Int

    /**
     * Connect to a BLE device and begin streaming ECG data.
     *
     * @param macAddress MAC address of the target device.
     * @param onData Callback invoked with each chunk of signal data.
     *               The FloatArray contains [leadCount] interleaved samples.
     * @param onError Callback invoked on connection or streaming errors.
     */
    suspend fun connect(
        macAddress: String,
        onData: (FloatArray) -> Unit,
        onError: (Exception) -> Unit
    )

    /** Disconnect from the device and stop streaming. */
    suspend fun disconnect()

    /** Whether the device is currently connected and streaming. */
    val isConnected: Boolean

    /**
     * Scan for nearby devices of this type.
     *
     * @param timeoutMs Scan timeout in milliseconds.
     * @return List of discovered device addresses.
     */
    suspend fun scan(timeoutMs: Long = 10_000): List<DiscoveredDevice>
}

/**
 * A BLE device discovered during scanning.
 */
data class DiscoveredDevice(
    val name: String?,
    val macAddress: String,
    val rssi: Int
)

/**
 * Stub implementation for AliveCor KardiaMobile-compatible devices.
 *
 * This is a placeholder — the actual BLE protocol for AliveCor devices
 * requires device-specific SDK integration. This stub defines the interface
 * contract that a real implementation would fulfill.
 */
class AliveCorDeviceStub : BleEcgDevice {

    override val deviceName: String = "AliveCor KardiaMobile"
    override val serviceUuid: String = "0000fff0-0000-1000-8000-00805f9b34fb"
    override val leadCount: Int = 1
    override val sampleRate: Int = 300

    override var isConnected: Boolean = false
        private set

    override suspend fun connect(
        macAddress: String,
        onData: (FloatArray) -> Unit,
        onError: (Exception) -> Unit
    ) {
        onError(UnsupportedOperationException(
            "AliveCor BLE integration requires device-specific SDK. " +
            "This is a protocol stub for the extensible device interface."
        ))
    }

    override suspend fun disconnect() {
        isConnected = false
    }

    override suspend fun scan(timeoutMs: Long): List<DiscoveredDevice> {
        // Stub: returns empty list
        return emptyList()
    }
}
