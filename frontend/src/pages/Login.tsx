import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './Login.css';

export function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();
  const { loginWithOAuth, loginWithApiKey, loginWithCredentials, isAuthenticated } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Redirect target after successful login
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/';

  // If already authenticated, redirect immediately
  if (isAuthenticated) {
    navigate(from, { replace: true });
    return null;
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    try {
      const success = await loginWithCredentials(email, password);
      if (success) {
        navigate(from, { replace: true });
      } else {
        setError(t('login.invalidEmail'));
      }
    } catch {
      setError(t('login.loginFailed'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOAuth = (provider: 'google' | 'github') => {
    loginWithOAuth(provider);
  };

  const handleApiKey = async () => {
    if (!apiKey.trim()) {
      setError(t('login.enterApiKey'));
      return;
    }
    setError('');
    setIsSubmitting(true);
    try {
      const success = await loginWithApiKey(apiKey.trim());
      if (success) {
        navigate(from, { replace: true });
      } else {
        setError(t('login.invalidApiKey'));
      }
    } catch {
      setError(t('login.apiKeyFailed'));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="login-page" id="page-login">
      {/* Background glow */}
      <div className="login-bg-glow" />

      <div className="login-container animate-fade-in">
        {/* Logo */}
        <div className="login-brand">
          <div className="login-logo">
            <span className="login-logo-icon">♥</span>
          </div>
          <h1 className="login-title">{t('login.title')}</h1>
          <p className="login-subtitle">{t('login.subtitle')}</p>
        </div>

        {/* Login card */}
        <div className="login-card glass">
          {error && (
            <div className="login-error" id="login-error-msg" role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label htmlFor="email-input" className="form-label">{t('login.email')}</label>
              <input
                type="email"
                id="email-input"
                className="form-input"
                placeholder="clinician@hospital.org"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="password-input" className="form-label">{t('login.password')}</label>
              <input
                type="password"
                id="password-input"
                className="form-input"
                placeholder="••••••••"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary login-submit-btn"
              id="login-submit-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? t('login.signingIn') : t('login.signIn')}
            </button>
          </form>

          <div className="login-divider">
            <span>{t('login.orContinueWith')}</span>
          </div>

          <div className="login-oauth">
            <button
              className="btn btn-secondary login-oauth-btn"
              id="oauth-google-btn"
              type="button"
              onClick={() => handleOAuth('google')}
              disabled={isSubmitting}
            >
              Google
            </button>
            <button
              className="btn btn-secondary login-oauth-btn"
              id="oauth-github-btn"
              type="button"
              onClick={() => handleOAuth('github')}
              disabled={isSubmitting}
            >
              GitHub
            </button>
          </div>

          {/* API key */}
          <div className="login-apikey">
            <details className="login-apikey-details">
              <summary className="login-apikey-summary">{t('login.useApiKey')}</summary>
              <div className="form-group" style={{ marginTop: 'var(--space-3)' }}>
                <input
                  type="text"
                  id="apikey-input"
                  className="form-input"
                  placeholder="ak_xxxxxxxxxxxxxxxxxxxx"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <button
                  className="btn btn-secondary login-apikey-submit"
                  id="apikey-submit-btn"
                  type="button"
                  onClick={handleApiKey}
                  disabled={isSubmitting}
                >
                  {t('login.authenticate')}
                </button>
              </div>
            </details>
          </div>
        </div>

        <p className="login-footer">
          {t('common.footer')}
        </p>
      </div>
    </div>
  );
}
