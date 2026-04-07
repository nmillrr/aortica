import { Link } from 'react-router-dom';
import './Batch.css';

const MOCK_BATCH = [
  { id: 'b-001', filename: 'patient_4821.hea', quality: 92, rhythm: 'AF',            structural: 'LVH',  riskMort: 0.12, riskHF: 0.28, status: 'done'    },
  { id: 'b-002', filename: 'patient_4820.csv', quality: 88, rhythm: 'NSR',           structural: 'Normal',riskMort: 0.04, riskHF: 0.06, status: 'done'    },
  { id: 'b-003', filename: 'patient_4819.dcm', quality: 76, rhythm: 'LBBB',          structural: 'LVH',  riskMort: 0.31, riskHF: 0.45, status: 'done'    },
  { id: 'b-004', filename: 'patient_4818.mat', quality: 95, rhythm: 'Sinus Brady',   structural: 'Normal',riskMort: 0.02, riskHF: 0.03, status: 'done'    },
  { id: 'b-005', filename: 'patient_4817.xml', quality: 61, rhythm: 'PVC',           structural: 'DCM',  riskMort: 0.55, riskHF: 0.62, status: 'done'    },
] as const;

export function Batch() {
  return (
    <div className="batch-page" id="page-batch">
      <div className="batch-toolbar">
        <div className="batch-toolbar-left">
          <Link to="/upload" className="btn btn-primary" id="batch-upload-btn">
            ↑ Upload Batch
          </Link>
          <span className="batch-count">{MOCK_BATCH.length} ECGs processed</span>
        </div>
        <button className="btn btn-secondary" id="batch-export-csv-btn">
          ⬇ Export CSV
        </button>
      </div>

      <div className="batch-table-container card" id="batch-results-table">
        <table className="batch-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Quality</th>
              <th>Top Rhythm</th>
              <th>Top Structural</th>
              <th>Mortality Risk</th>
              <th>HF Risk</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_BATCH.map(row => (
              <tr key={row.id} className="batch-row">
                <td>
                  <Link to={`/results/${row.id}`} className="batch-filename" id={`batch-row-${row.id}`}>
                    {row.filename}
                  </Link>
                </td>
                <td>
                  <span className={`quality-pill ${row.quality >= 70 ? 'quality-pill--good' : row.quality >= 40 ? 'quality-pill--marginal' : 'quality-pill--poor'}`}>
                    {row.quality}
                  </span>
                </td>
                <td className="batch-finding">{row.rhythm}</td>
                <td className="batch-finding">{row.structural}</td>
                <td>
                  <span className={`risk-cell ${row.riskMort > 0.4 ? 'risk-cell--high' : row.riskMort > 0.2 ? 'risk-cell--medium' : 'risk-cell--low'}`}>
                    {(row.riskMort * 100).toFixed(0)}%
                  </span>
                </td>
                <td>
                  <span className={`risk-cell ${row.riskHF > 0.4 ? 'risk-cell--high' : row.riskHF > 0.2 ? 'risk-cell--medium' : 'risk-cell--low'}`}>
                    {(row.riskHF * 100).toFixed(0)}%
                  </span>
                </td>
                <td>
                  <span className="status-pill status-pill--done">Done</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
