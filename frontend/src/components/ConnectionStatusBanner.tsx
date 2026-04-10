import { useState, useEffect, useCallback } from 'react';
import { checkServerHealth } from '../services/InferenceClient';
import './ConnectionStatusBanner.css';

export type ConnectionStatus = 'server' | 'offline' | 'checking';

interface ConnectionStatusBannerProps {
  /** Override the polling interval in ms (default: 10000) */
  pollIntervalMs?: number;
  /** Override the server URL to check */
  serverUrl?: string;
}

/**
 * ConnectionStatusBanner — Shows connection state in the sidebar footer.
 *
 * - Green badge: "Server — Full Model" when local FastAPI responds to /health
 * - Amber badge: "Offline Mode — Edge Model" when using WASM fallback
 * - Pulsing dot while checking connectivity
 */
export function ConnectionStatusBanner({
  pollIntervalMs = 10_000,
  serverUrl,
}: ConnectionStatusBannerProps) {
  const [status, setStatus] = useState<ConnectionStatus>('checking');

  const checkConnection = useCallback(async () => {
    const isOnline = await checkServerHealth(serverUrl);
    setStatus(isOnline ? 'server' : 'offline');
  }, [serverUrl]);

  useEffect(() => {
    // Initial check
    void checkConnection();

    // Poll periodically
    const interval = setInterval(() => {
      void checkConnection();
    }, pollIntervalMs);

    return () => clearInterval(interval);
  }, [checkConnection, pollIntervalMs]);

  return (
    <div
      className={`connection-banner connection-banner--${status}`}
      id="connection-status-banner"
      role="status"
      aria-live="polite"
    >
      <span className={`connection-dot connection-dot--${status}`} />
      <div className="connection-info">
        <span className="connection-label">
          {status === 'server' && 'Server — Full Model'}
          {status === 'offline' && 'Offline Mode — Edge Model'}
          {status === 'checking' && 'Checking connection…'}
        </span>
        <span className="connection-detail">
          {status === 'server' && 'Connected to local server'}
          {status === 'offline' && 'Using in-browser WASM inference'}
          {status === 'checking' && 'Attempting server contact'}
        </span>
      </div>
    </div>
  );
}
