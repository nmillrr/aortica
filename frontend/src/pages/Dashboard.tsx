import { Link } from 'react-router-dom';
import './Dashboard.css';

const STATS = [
  { label: 'ECGs Analyzed',    value: '1,247',  change: '+12%',  trend: 'up'   },
  { label: 'Flagged Findings', value: '89',     change: '+3%',   trend: 'up'   },
  { label: 'Edge Inferences',  value: '392',    change: '+24%',  trend: 'up'   },
  { label: 'Avg Confidence',   value: '94.2%',  change: '+0.8%', trend: 'up'   },
] as const;

const RECENT = [
  { id: 'ecg-001', patient: 'Patient #4821', time: '2 min ago',  finding: 'AF detected',         severity: 'high'   },
  { id: 'ecg-002', patient: 'Patient #4820', time: '15 min ago', finding: 'Normal Sinus Rhythm',  severity: 'normal' },
  { id: 'ecg-003', patient: 'Patient #4819', time: '43 min ago', finding: 'LBBB',                 severity: 'medium' },
  { id: 'ecg-004', patient: 'Patient #4818', time: '1 hr ago',   finding: 'LVH suspected',        severity: 'medium' },
  { id: 'ecg-005', patient: 'Patient #4817', time: '2 hr ago',   finding: 'Normal Sinus Rhythm',  severity: 'normal' },
] as const;

const TASKS = [
  { name: 'Rhythm',     classes: 22, icon: '♥' },
  { name: 'Structural', classes: 15, icon: '◇' },
  { name: 'Ischaemia',  classes: 10, icon: '△' },
  { name: 'Risk',       classes: 3,  icon: '⚡' },
] as const;

export function Dashboard() {
  return (
    <div className="dashboard" id="page-dashboard">
      {/* Hero welcome */}
      <section className="dashboard-hero">
        <div className="dashboard-hero-text">
          <h2 className="dashboard-hero-title">
            Welcome to <span className="text-accent">Aortica</span>
          </h2>
          <p className="dashboard-hero-subtitle">
            AI-powered ECG analysis platform — multi-task deep learning with explainable predictions.
          </p>
          <Link to="/upload" className="btn btn-primary" id="hero-upload-btn">
            ↑ Upload ECG
          </Link>
        </div>
      </section>

      {/* Stats */}
      <section className="dashboard-stats" id="stats-grid">
        {STATS.map(stat => (
          <div className="stat-card card" key={stat.label}>
            <span className="stat-label">{stat.label}</span>
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
            <h3 className="card-title">Recent ECGs</h3>
            <Link to="/batch" className="card-action btn-ghost btn">View all</Link>
          </div>
          <ul className="recent-list">
            {RECENT.map(ecg => (
              <li key={ecg.id} className="recent-item">
                <Link to={`/results/${ecg.id}`} className="recent-link" id={`recent-${ecg.id}`}>
                  <div className="recent-info">
                    <span className={`recent-severity-dot severity-${ecg.severity}`} />
                    <div>
                      <span className="recent-patient">{ecg.patient}</span>
                      <span className="recent-finding">{ecg.finding}</span>
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
          <h3 className="dashboard-section-title">Multi-Task Engine</h3>
          <div className="task-cards">
            {TASKS.map(task => (
              <div className="task-card card" key={task.name}>
                <span className="task-icon">{task.icon}</span>
                <span className="task-name">{task.name}</span>
                <span className="task-classes">{task.classes} classes</span>
              </div>
            ))}
          </div>
          <div className="model-badge">
            <span className="model-badge-dot" />
            <span>Model v0.2.0 — full precision</span>
          </div>
        </section>
      </div>
    </div>
  );
}
