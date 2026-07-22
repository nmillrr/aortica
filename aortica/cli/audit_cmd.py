"""``aortica audit`` — audit-trail export and verification (US-121).

\b
Subcommands:
  aortica audit export --db <path> --from <date> --to <date> --format csv|json
  aortica audit verify --db <path>
"""

from __future__ import annotations

import csv
import io
import json
import sys
from typing import Optional

import click


@click.group(name="audit")
def audit_group() -> None:
    """Audit-trail commands."""


@audit_group.command(name="export")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True),
              help="Path to the audit SQLite database.")
@click.option("--from", "date_from", default=None, help="ISO start date (inclusive).")
@click.option("--to", "date_to", default=None, help="ISO end date (inclusive).")
@click.option("--format", "output_format",
              type=click.Choice(["csv", "json"], case_sensitive=False),
              default="csv", show_default=True, help="Export format.")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Write to this file instead of stdout.")
@click.option("--hmac-key", default=None, help="HMAC key (else env AORTICA_AUDIT_HMAC_KEY).")
def export_cmd(
    db_path: str,
    date_from: Optional[str],
    date_to: Optional[str],
    output_format: str,
    output_path: Optional[str],
    hmac_key: Optional[str],
) -> None:
    """Export audit events for a date range (for regulatory submission)."""
    from aortica.audit import AuditLogger

    logger = AuditLogger(db_path, hmac_key=hmac_key)
    events = logger.query(date_from=date_from, date_to=date_to, limit=1_000_000)
    # Oldest-first for a readable regulatory export.
    events = list(reversed(events))

    if output_format == "json":
        text = json.dumps([e.to_dict() for e in events], indent=2)
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "event_type", "user_id", "ecg_reference_id",
            "model_version", "session_id", "ip_address", "event_details", "hmac",
        ])
        for e in events:
            writer.writerow([
                e.id, e.timestamp, e.event_type, e.user_id or "",
                e.ecg_reference_id or "", e.model_version or "",
                e.session_id or "", e.ip_address or "",
                json.dumps(e.event_details, sort_keys=True), e.hmac,
            ])
        text = buf.getvalue()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        click.echo(f"Exported {len(events)} event(s) to {output_path}")
    else:
        click.echo(text)


@audit_group.command(name="verify")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True),
              help="Path to the audit SQLite database.")
@click.option("--hmac-key", default=None, help="HMAC key (else env AORTICA_AUDIT_HMAC_KEY).")
def verify_cmd(db_path: str, hmac_key: Optional[str]) -> None:
    """Verify the audit log's HMAC chain integrity."""
    from aortica.audit import verify_integrity

    report = verify_integrity(db_path, hmac_key=hmac_key)
    click.echo(json.dumps(report.to_dict(), indent=2))
    if not report.valid:
        sys.exit(1)
