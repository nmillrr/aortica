import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import './ModelCompare.css';

// ---------------------------------------------------------------------------
// Types — mirror aortica.evaluation.benchmark.BenchmarkReport.as_dict()
// ---------------------------------------------------------------------------

interface ClassMetrics {
  name: string;
  auc: number;
  sensitivity: number;
  specificity: number;
  f1: number;
}

interface TaskReport {
  task_name: string;
  macro_f1: number;
  ece: number;
  c_index: number;
  brier_score: number;
  per_class: ClassMetrics[];
}

interface BenchmarkReport {
  overall: Record<string, TaskReport>;
  n_samples: number;
  tasks_evaluated: string[];
}

const CLASSIFICATION_TASKS = ['rhythm', 'structural', 'ischaemia'];

// A delta this small (or smaller) is treated as noise, not a real change.
const DELTA_EPSILON = 1e-4;

interface ClassDeltaRow {
  name: string;
  a: ClassMetrics;
  b: ClassMetrics;
  deltaF1: number;
  deltaAuc: number;
  deltaSens: number;
  deltaSpec: number;
  regression: boolean;
}

interface TaskDeltaView {
  task: string;
  deltaMacroF1: number;
  deltaCIndex: number;
  classes: ClassDeltaRow[];
}

// ---------------------------------------------------------------------------
// Delta helpers
// ---------------------------------------------------------------------------

function deltaClass(delta: number): string {
  if (delta > DELTA_EPSILON) return 'delta-up';
  if (delta < -DELTA_EPSILON) return 'delta-down';
  return 'delta-flat';
}

function fmtDelta(delta: number): string {
  const sign = delta > 0 ? '+' : '';
  return `${sign}${delta.toFixed(4)}`;
}

function computeTaskViews(a: BenchmarkReport, b: BenchmarkReport): TaskDeltaView[] {
  const tasks = a.tasks_evaluated.filter((t) => b.tasks_evaluated.includes(t));
  return tasks.map((task) => {
    const ta = a.overall[task];
    const tb = b.overall[task];
    const classesA = ta?.per_class ?? [];
    const classesB = tb?.per_class ?? [];
    const byName = new Map(classesB.map((c) => [c.name, c]));

    const classes: ClassDeltaRow[] = classesA.map((ca) => {
      const cb = byName.get(ca.name) ?? ca;
      const deltaF1 = cb.f1 - ca.f1;
      return {
        name: ca.name,
        a: ca,
        b: cb,
        deltaF1,
        deltaAuc: cb.auc - ca.auc,
        deltaSens: cb.sensitivity - ca.sensitivity,
        deltaSpec: cb.specificity - ca.specificity,
        regression: deltaF1 < -DELTA_EPSILON,
      };
    });

    return {
      task,
      deltaMacroF1: (tb?.macro_f1 ?? 0) - (ta?.macro_f1 ?? 0),
      deltaCIndex: (tb?.c_index ?? 0) - (ta?.c_index ?? 0),
      classes,
    };
  });
}

function recommend(views: TaskDeltaView[]): 'upgrade' | 'hold' | 'investigate' {
  const regressions = views.some((v) => v.classes.some((c) => c.regression));
  if (regressions) return 'investigate';
  const net = views.reduce((acc, v) => acc + v.deltaMacroF1 + v.deltaCIndex, 0);
  return net > DELTA_EPSILON ? 'upgrade' : 'hold';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ModelComparePage() {
  const { t } = useTranslation();
  const [reportA, setReportA] = useState<BenchmarkReport | null>(null);
  const [reportB, setReportB] = useState<BenchmarkReport | null>(null);
  const [labelA, setLabelA] = useState('Model A');
  const [labelB, setLabelB] = useState('Model B');
  const [error, setError] = useState<string | null>(null);

  const parseFile = (
    file: File,
    setter: (r: BenchmarkReport) => void,
    setLabel: (s: string) => void,
  ) => {
    setError(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result as string) as BenchmarkReport;
        if (!parsed.overall || !parsed.tasks_evaluated) {
          throw new Error('Not a benchmark report (missing overall/tasks_evaluated).');
        }
        setter(parsed);
        setLabel(file.name.replace(/\.json$/i, ''));
      } catch (e) {
        setError(`Failed to parse ${file.name}: ${(e as Error).message}`);
      }
    };
    reader.readAsText(file);
  };

  const views = useMemo(
    () => (reportA && reportB ? computeTaskViews(reportA, reportB) : []),
    [reportA, reportB],
  );

  const recommendation = useMemo(
    () => (views.length ? recommend(views) : null),
    [views],
  );

  const totalRegressions = useMemo(
    () => views.reduce((acc, v) => acc + v.classes.filter((c) => c.regression).length, 0),
    [views],
  );

  return (
    <div className="model-compare" id="model-compare-page">
      <header className="mc-header">
        <h1>{t('compare.title', 'Model Version Comparison')}</h1>
        <p className="mc-subtitle">
          {t(
            'compare.subtitle',
            'Upload two benchmark reports (JSON) to compare model versions side-by-side.',
          )}
        </p>
      </header>

      <section className="mc-uploads">
        <div className="mc-upload-card">
          <label className="mc-upload-label" htmlFor="mc-file-a">
            {t('compare.uploadA', 'Benchmark report A')}
          </label>
          <input
            id="mc-file-a"
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) parseFile(f, setReportA, setLabelA);
            }}
          />
          {reportA && (
            <span className="mc-file-ok">
              ✓ {labelA} ({reportA.n_samples} samples)
            </span>
          )}
        </div>

        <div className="mc-upload-card">
          <label className="mc-upload-label" htmlFor="mc-file-b">
            {t('compare.uploadB', 'Benchmark report B')}
          </label>
          <input
            id="mc-file-b"
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) parseFile(f, setReportB, setLabelB);
            }}
          />
          {reportB && (
            <span className="mc-file-ok">
              ✓ {labelB} ({reportB.n_samples} samples)
            </span>
          )}
        </div>
      </section>

      {error && <div className="mc-error" role="alert">{error}</div>}

      {recommendation && (
        <section className={`mc-recommendation mc-rec-${recommendation}`}>
          <span className="mc-rec-badge">{recommendation.toUpperCase()}</span>
          <span className="mc-rec-detail">
            {totalRegressions > 0
              ? t('compare.regressionsFound', {
                  defaultValue: '{{count}} class regression(s) detected',
                  count: totalRegressions,
                })
              : t('compare.noRegressions', 'No regressions detected')}
          </span>
        </section>
      )}

      {views.length > 0 && (
        <section className="mc-summary">
          <h2>{t('compare.summary', 'Summary')}</h2>
          <table className="mc-table">
            <thead>
              <tr>
                <th>{t('compare.task', 'Task')}</th>
                <th>Δ Macro-F1</th>
                <th>Δ C-index</th>
                <th>{t('compare.regressions', 'Regressions')}</th>
              </tr>
            </thead>
            <tbody>
              {views.map((v) => (
                <tr key={v.task}>
                  <td>{v.task}</td>
                  <td className={deltaClass(v.deltaMacroF1)}>
                    {CLASSIFICATION_TASKS.includes(v.task) ? fmtDelta(v.deltaMacroF1) : '—'}
                  </td>
                  <td className={deltaClass(v.deltaCIndex)}>
                    {v.task === 'risk' ? fmtDelta(v.deltaCIndex) : '—'}
                  </td>
                  <td>{v.classes.filter((c) => c.regression).length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {views.map((v) => (
        <section key={v.task} className="mc-task-detail">
          <h2>{v.task}</h2>
          <div className="mc-table-scroll">
            <table className="mc-table">
              <thead>
                <tr>
                  <th>{t('compare.class', 'Class')}</th>
                  <th>Δ F1</th>
                  <th>Δ AUC</th>
                  <th>Δ Sens</th>
                  <th>Δ Spec</th>
                </tr>
              </thead>
              <tbody>
                {v.classes.map((c) => (
                  <tr key={c.name} className={c.regression ? 'mc-row-regression' : ''}>
                    <td>
                      {c.regression && <span className="mc-flag">🔴</span>} {c.name}
                    </td>
                    <td className={deltaClass(c.deltaF1)}>{fmtDelta(c.deltaF1)}</td>
                    <td className={deltaClass(c.deltaAuc)}>{fmtDelta(c.deltaAuc)}</td>
                    <td className={deltaClass(c.deltaSens)}>{fmtDelta(c.deltaSens)}</td>
                    <td className={deltaClass(c.deltaSpec)}>{fmtDelta(c.deltaSpec)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {!reportA || !reportB ? (
        <p className="mc-hint">
          {t('compare.hint', 'Generate reports with: aortica benchmark <dataset> --format json')}
        </p>
      ) : null}
    </div>
  );
}
