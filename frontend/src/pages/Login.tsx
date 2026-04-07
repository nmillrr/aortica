import { useNavigate } from 'react-router-dom';
import './Login.css';

export function Login() {
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/');
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
          <h1 className="login-title">Aortica</h1>
          <p className="login-subtitle">AI-Powered ECG Analysis Platform</p>
        </div>

        {/* Login card */}
        <div className="login-card glass">
          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label htmlFor="email-input" className="form-label">Email</label>
              <input
                type="email"
                id="email-input"
                className="form-input"
                placeholder="clinician@hospital.org"
                autoComplete="email"
              />
            </div>
            <div className="form-group">
              <label htmlFor="password-input" className="form-label">Password</label>
              <input
                type="password"
                id="password-input"
                className="form-input"
                placeholder="••••••••"
                autoComplete="current-password"
              />
            </div>
            <button type="submit" className="btn btn-primary login-submit-btn" id="login-submit-btn">
              Sign In
            </button>
          </form>

          <div className="login-divider">
            <span>or continue with</span>
          </div>

          <div className="login-oauth">
            <button className="btn btn-secondary login-oauth-btn" id="oauth-google-btn" type="button">
              Google
            </button>
            <button className="btn btn-secondary login-oauth-btn" id="oauth-github-btn" type="button">
              GitHub
            </button>
          </div>

          {/* API key */}
          <div className="login-apikey">
            <details className="login-apikey-details">
              <summary className="login-apikey-summary">Use API Key</summary>
              <div className="form-group" style={{ marginTop: 'var(--space-3)' }}>
                <input
                  type="text"
                  id="apikey-input"
                  className="form-input"
                  placeholder="ak_xxxxxxxxxxxxxxxxxxxx"
                />
                <button className="btn btn-secondary login-apikey-submit" id="apikey-submit-btn" type="button">
                  Authenticate
                </button>
              </div>
            </details>
          </div>
        </div>

        <p className="login-footer">
          Self-hosted · No data leaves your deployment
        </p>
      </div>
    </div>
  );
}
