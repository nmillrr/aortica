import { Link } from 'react-router-dom';
import { Trans, useTranslation } from 'react-i18next';
import './Dashboard.css';

const STATS = [
  { key: 'ecgsAnalyzed',    value: '1,247',  change: '+12%',  trend: 'up'   },
  { key: 'flaggedFindings', value: '89',     change: '+3%',   trend: 'up'   },
  { key: 'edgeInferences',  value: '392',    change: '+24%',  trend: 'up'   },
  { key: 'avgConfidence',   value: '94.2%',  change: '+0.8%', trend: 'up'   },
] as const;

const RECENT = [
  { id: 'ecg-001', patient: 4821, time: '2 min ago',  finding: 'AF',                  severity: 'high'   },
  { id: 'ecg-002', patient: 4820, time: '15 min ago', finding: 'Normal Sinus Rhythm', severity: 'normal' },
  { id: 'ecg-003', patient: 4819, time: '43 min ago', finding: 'LBBB',                severity: 'medium' },
  { id: 'ecg-004', patient: 4818, time: '1 hr ago',   finding: 'LVH',                 severity: 'medium' },
  { id: 'ecg-005', patient: 4817, time: '2 hr ago',   finding: 'Normal Sinus Rhythm', severity: 'normal' },
] as const;

const TASKS = [
  { key: 'rhythm',     classes: 22, icon: '♥' },
  { key: 'structural', classes: 15, icon: '◇' },
  { key: 'ischaemia',  classes: 10, icon: '△' },
  { key: 'risk',       classes: 3,  icon: '⚡' },
] as const;

export function Dashboard() {
  const { t, i18n } = useTranslation();
  const numberFmt = new Intl.NumberFormat(i18n.resolvedLanguage);

  return (
    <div className="dashboard" id="page-dashboard">
      {/* Hero welcome */}
      <section className="dashboard-hero">
        <div className="dashboard-hero-text">
          <h2 className="dashboard-hero-title">
            <Trans i18nKey="dashboard.heroTitle" components={{ accent: <span className="text-accent" /> }} />
          </h2>
          <p className="dashboard-hero-subtitle">
            {t('dashboard.heroSubtitle')}
          </p>
          <Link to="/upload" className="btn btn-primary" id="hero-upload-btn">
            {t('dashboard.uploadBtn')}
          </Link>
        </div>
      </section>

      {/* Stats */}
      <section className="dashboard-stats" id="stats-grid">
        {STATS.map(stat => (
          <div className="stat-card card" key={stat.key}>
            <span className="stat-label">{t(`dashboard.stats.${stat.key}`)}</span>
            <span className="stat-value">{stat.value}</span>
            <span className={`stat-change stat-change--${stat.trend}`}>
              {stat.change}
            </span>
          </div>
        ))}
      </section>

      {/* Two columns: recent ECGs + task heads */}
      <div className="dashboard-grid">
        {/* Recent */}
        <section className="dashboard-recent card" id="recent-ecgs">
          <div className="card-header">
            <h3 className="card-title">{t('dashboard.recentEcgs')}</h3>
            <Link to="/batch" className="card-action btn-ghost btn">{t('common.viewAll')}</Link>
          </div>
          <ul className="recent-list">
            {RECENT.map(ecg => (
              <li key={ecg.id} className="recent-item">
                <Link to={`/results/${ecg.id}`} className="recent-link" id={`recent-${ecg.id}`}>
                  <div className="recent-info">
                    <span className={`recent-severity-dot severity-${ecg.severity}`} />
                    <div>
                      <span className="recent-patient">
                        {t('dashboard.patient', 'Patient #{{id}}', { id: numberFmt.format(ecg.patient) })}
                      </span>
                      <span className="recent-finding">{t(`conditions.${ecg.finding}`, ecg.finding)}</span>
                    </div>
                  </div>
                  <span className="recent-time">{ecg.time}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>

        {/* Task heads overview */}
        <section className="dashboard-tasks" id="task-heads-overview">
          <h3 className="dashboard-section-title">{t('dashboard.multiTaskEngine')}</h3>
          <div className="task-cards">
            {TASKS.map(task => (
              <div className="task-card card" key={task.key}>
                <span className="task-icon">{task.icon}</span>
                <span className="task-name">{t(`dashboard.tasks.${task.key}`)}</span>
                <span className="task-classes">{t('dashboard.classes', { count: task.classes })}</span>
              </div>
            ))}
          </div>
          <div className="model-badge">
            <span className="model-badge-dot" />
            <span>{t('dashboard.modelBadge')}</span>
          </div>
        </section>
      </div>
    </div>
  );
}
