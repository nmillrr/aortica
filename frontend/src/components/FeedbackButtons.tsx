import { useState, useCallback } from 'react';
import './FeedbackButtons.css';

/* ---------- Types -------------------------------------------------------- */

export interface FeedbackAction {
  findingName: string;
  task: string;
  action: 'accept' | 'reject' | 'modify';
  comment?: string;
  confidence: number;
}

interface FeedbackButtonsProps {
  findingName: string;
  task: string;
  confidence: number;
  /** ECG reference ID for the current analysis session. */
  ecgReferenceId?: string;
  /** Called after feedback is submitted (or locally recorded). */
  onFeedbackSubmit?: (feedback: FeedbackAction) => void;
}

/* ---------- Component ---------------------------------------------------- */

export function FeedbackButtons({
  findingName,
  task,
  confidence,
  ecgReferenceId,
  onFeedbackSubmit,
}: FeedbackButtonsProps) {
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [showModify, setShowModify] = useState(false);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitFeedback = useCallback(
    async (action: 'accept' | 'reject' | 'modify', feedbackComment?: string) => {
      setIsSubmitting(true);

      const payload = {
        ecg_reference_id: ecgReferenceId || `session-${Date.now()}`,
        finding_name: findingName,
        task,
        action,
        comment: feedbackComment || undefined,
        ai_confidence: confidence,
      };

      // Try server first, fall back to local recording
      try {
        const response = await fetch('/api/v1/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error('Server error');
      } catch {
        // Offline or server unavailable — record locally
        const stored = JSON.parse(localStorage.getItem('aortica_pending_feedback') || '[]');
        stored.push({ ...payload, timestamp: new Date().toISOString() });
        localStorage.setItem('aortica_pending_feedback', JSON.stringify(stored));
      }

      setSubmitted(action);
      setIsSubmitting(false);
      setShowModify(false);

      onFeedbackSubmit?.({
        findingName,
        task,
        action,
        comment: feedbackComment,
        confidence,
      });
    },
    [findingName, task, confidence, ecgReferenceId, onFeedbackSubmit],
  );

  /* ---- Already submitted state ---- */
  if (submitted) {
    return (
      <div className="feedback-buttons feedback-buttons--submitted" id={`feedback-${findingName.replace(/\s+/g, '-')}`}>
        <span className={`feedback-badge feedback-badge--${submitted}`}>
          {submitted === 'accept' && '✓ Accepted'}
          {submitted === 'reject' && '✗ Rejected'}
          {submitted === 'modify' && '✎ Modified'}
        </span>
        <button
          className="feedback-undo"
          onClick={() => { setSubmitted(null); setShowModify(false); setComment(''); }}
          title="Change feedback"
        >
          ↩
        </button>
      </div>
    );
  }

  /* ---- Modify form ---- */
  if (showModify) {
    return (
      <div className="feedback-buttons feedback-buttons--modify" id={`feedback-${findingName.replace(/\s+/g, '-')}`}>
        <textarea
          className="feedback-comment"
          placeholder="Describe your modification…"
          value={comment}
          onChange={e => setComment(e.target.value)}
          rows={2}
          id={`feedback-comment-${findingName.replace(/\s+/g, '-')}`}
        />
        <div className="feedback-modify-actions">
          <button
            className="feedback-btn feedback-btn--modify-submit"
            onClick={() => submitFeedback('modify', comment)}
            disabled={isSubmitting || comment.trim().length === 0}
          >
            {isSubmitting ? '…' : 'Submit'}
          </button>
          <button
            className="feedback-btn feedback-btn--cancel"
            onClick={() => { setShowModify(false); setComment(''); }}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  /* ---- Default buttons ---- */
  return (
    <div className="feedback-buttons" id={`feedback-${findingName.replace(/\s+/g, '-')}`}>
      <button
        className="feedback-btn feedback-btn--accept"
        onClick={() => submitFeedback('accept')}
        disabled={isSubmitting}
        title="Accept this finding"
      >
        ✓
      </button>
      <button
        className="feedback-btn feedback-btn--reject"
        onClick={() => submitFeedback('reject')}
        disabled={isSubmitting}
        title="Reject this finding"
      >
        ✗
      </button>
      <button
        className="feedback-btn feedback-btn--modify"
        onClick={() => setShowModify(true)}
        disabled={isSubmitting}
        title="Modify this finding"
      >
        ✎
      </button>
    </div>
  );
}
