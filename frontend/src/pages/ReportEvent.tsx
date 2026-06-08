import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { AdverseEventForm } from '../components/AdverseEventForm';
import './ReportEvent.css';

export function ReportEvent() {
  const navigate = useNavigate();

  const handleSubmitted = useCallback(
    (_eventId: string) => {
      // Stay on form — success state is shown in the component
    },
    [],
  );

  const handleCancel = useCallback(() => {
    navigate('/');
  }, [navigate]);

  return (
    <div className="report-event-page" id="report-event-page">
      <AdverseEventForm
        onSubmitted={handleSubmitted}
        onCancel={handleCancel}
      />
    </div>
  );
}
