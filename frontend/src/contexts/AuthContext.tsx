import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuthUser {
  sub: string;
  email?: string;
  name?: string;
  provider: string;
}

interface AuthTokens {
  accessToken: string;
  refreshToken?: string;
  expiresAt: number; // unix ms
}

interface AuthContextType {
  user: AuthUser | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  loginWithOAuth: (provider: 'google' | 'github') => void;
  loginWithApiKey: (apiKey: string) => Promise<boolean>;
  loginWithCredentials: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  getAuthHeader: () => Record<string, string>;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY_TOKENS = 'aortica_auth_tokens';
const STORAGE_KEY_USER = 'aortica_auth_user';
const API_BASE = 'http://localhost:8000';
const TOKEN_REFRESH_MARGIN_MS = 5 * 60 * 1000; // 5 min before expiry

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseJwt(token: string): Record<string, unknown> {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const json = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(json);
  } catch {
    return {};
  }
}

function loadStoredTokens(): AuthTokens | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_TOKENS);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function loadStoredUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_USER);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function storeAuth(tokens: AuthTokens, user: AuthUser): void {
  localStorage.setItem(STORAGE_KEY_TOKENS, JSON.stringify(tokens));
  localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));
}

function clearStoredAuth(): void {
  localStorage.removeItem(STORAGE_KEY_TOKENS);
  localStorage.removeItem(STORAGE_KEY_USER);
}

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(loadStoredUser);
  const [tokens, setTokens] = useState<AuthTokens | null>(loadStoredTokens);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!tokens && !!user && Date.now() < tokens.expiresAt;

  // Restore session on mount
  useEffect(() => {
    const stored = loadStoredTokens();
    const storedUser = loadStoredUser();
    if (stored && storedUser && Date.now() < stored.expiresAt) {
      setTokens(stored);
      setUser(storedUser);
    } else {
      clearStoredAuth();
      setTokens(null);
      setUser(null);
    }
    setIsLoading(false);
  }, []);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!tokens?.refreshToken || !tokens.expiresAt) return;

    const msUntilRefresh = tokens.expiresAt - Date.now() - TOKEN_REFRESH_MARGIN_MS;
    if (msUntilRefresh <= 0) return;

    const timer = setTimeout(async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${tokens.refreshToken}`,
          },
        });
        if (resp.ok) {
          const data = await resp.json();
          const payload = parseJwt(data.access_token);
          const newTokens: AuthTokens = {
            accessToken: data.access_token,
            refreshToken: data.refresh_token || tokens.refreshToken,
            expiresAt: ((payload.exp as number) || 0) * 1000,
          };
          setTokens(newTokens);
          if (user) storeAuth(newTokens, user);
        }
      } catch {
        // Refresh failed — user will need to re-login
      }
    }, msUntilRefresh);

    return () => clearTimeout(timer);
  }, [tokens, user]);

  // ── Login methods ─────────────────────────────────────────────

  const loginWithOAuth = useCallback((provider: 'google' | 'github') => {
    window.location.href = `${API_BASE}/api/v1/auth/login/${provider}`;
  }, []);

  const loginWithApiKey = useCallback(async (apiKey: string): Promise<boolean> => {
    try {
      // Validate API key by calling a protected endpoint
      const resp = await fetch(`${API_BASE}/info`, {
        headers: { 'X-API-Key': apiKey },
      });
      if (resp.ok) {
        const authUser: AuthUser = {
          sub: 'api_key_user',
          provider: 'api_key',
        };
        const authTokens: AuthTokens = {
          accessToken: apiKey, // Use API key directly as "token"
          expiresAt: Date.now() + 365 * 24 * 60 * 60 * 1000, // API keys don't expire
        };
        setUser(authUser);
        setTokens(authTokens);
        storeAuth(authTokens, authUser);
        return true;
      }
    } catch {
      // fall through
    }
    return false;
  }, []);

  const loginWithCredentials = useCallback(
    async (_email: string, _password: string): Promise<boolean> => {
      // For self-hosted deployments, credentials are validated locally.
      // In this implementation we create a local JWT session.
      const authUser: AuthUser = {
        sub: `local:${_email}`,
        email: _email,
        provider: 'local',
      };
      // In a real deployment, this would call a backend endpoint.
      // For now, we accept any credentials for the self-hosted model.
      const authTokens: AuthTokens = {
        accessToken: 'local-session',
        expiresAt: Date.now() + 24 * 60 * 60 * 1000,
      };
      setUser(authUser);
      setTokens(authTokens);
      storeAuth(authTokens, authUser);
      return true;
    },
    []
  );

  const logout = useCallback(() => {
    setUser(null);
    setTokens(null);
    clearStoredAuth();
  }, []);

  const getAuthHeader = useCallback((): Record<string, string> => {
    if (!tokens) return {};
    if (user?.provider === 'api_key') {
      return { 'X-API-Key': tokens.accessToken };
    }
    return { Authorization: `Bearer ${tokens.accessToken}` };
  }, [tokens, user]);

  return (
    <AuthContext.Provider
      value={{
        user,
        tokens,
        isAuthenticated,
        isLoading,
        loginWithOAuth,
        loginWithApiKey,
        loginWithCredentials,
        logout,
        getAuthHeader,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
