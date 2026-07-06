# LMIC Pilot Deployment Guide

This guide walks a field deployment engineer through standing up an Aortica edge
site at a rural or low- and middle-income-country (LMIC) pilot location. It
assembles the edge model (US-038–US-041), the Raspberry Pi profile (US-059), and
the community-health-worker output layer (US-060) into a deployable kit.

For the companion training material aimed at the operator, see
[CHW Training](CHW_TRAINING.md). For the go/no-go sign-off, see the
[Pilot Checklist](PILOT_CHECKLIST.md).

---

## 1. Site prerequisites

Confirm all three before travelling to the site.

### Power

- A mains supply **or** a battery/UPS that covers a full clinic day.
- The Raspberry Pi 4 draws ~4 W under load; with duty-cycled inference
  (see below) sustained draw attributable to analysis stays **under 200 mW**,
  so a modest power bank comfortably covers a day of intermittent use.
- A surge protector is strongly recommended where mains power is unstable.

### Connectivity

- **Intermittent connectivity is fine.** Aortica is offline-first: results are
  stored locally (encrypted) and sync to the central server whenever a
  connection is available.
- A 3G/4G dongle, shared Wi-Fi, or a periodic "sync run" to a connected location
  all work. No always-on link is required.

### Hardware

| Item | Requirement |
|------|-------------|
| Compute | Raspberry Pi 4, 4 GB RAM (or newer) |
| Storage | SD card, 32 GB or larger (A1/A2 rated) |
| ECG capture | Compatible serial/USB ECG front-end or handheld device |
| Power | Mains + battery/UPS covering clinic hours |
| Enclosure | Dust- and heat-tolerant case with ventilation |

---

## 2. SD card image preparation

The pilot image is produced with the repository's
[`create_pi_image_script.sh`](https://github.com/nmillrr/aortica/blob/main/create_pi_image_script.sh)
(from US-059), which installs Aortica, the edge model, and the systemd service.

```bash
# On a workstation with the SD card mounted:
sudo ./create_pi_image_script.sh --device /dev/sdX --profile rpi4

# Verify the written image before removing the card:
sudo ./create_pi_image_script.sh --verify --device /dev/sdX
```

The script provisions:

- Aortica (`pip install aortica[cli,edge]`) and the INT8 edge model.
- The `aortica-edge` systemd unit (auto-start, watchdog, restart-on-failure).
- Duty-cycled inference configuration (model loads on-demand per ECG).
- The data directory (`/var/lib/aortica/data`) and log directory
  (`/var/log/aortica`).

!!! tip "Keep a golden image"
    Flash **two** identical cards. Keep one sealed as a recovery image so a
    corrupted card can be swapped in the field without re-running the script.

---

## 3. First-boot walkthrough

1. **Insert the prepared SD card** and connect power. The green activity LED
   flickers during boot; first boot expands the filesystem and can take
   2–3 minutes.
2. **Wait for the service to come up.** The `aortica-edge` unit starts
   automatically. Confirm from an attached keyboard/monitor or over SSH:

    ```bash
    systemctl status aortica-edge      # should report "active (running)"
    ```

3. **Validate the edge model** (reuses US-041 validation and the US-061b power
   check):

    ```bash
    aortica edge site-report            # confirms the site DB and status
    ```

    *(Screenshot: the site report showing 0 inferences, storage healthy, sync
    "unknown" until the first upload.)*

4. **Run one test ECG** through the capture device and confirm a colour tier is
   displayed (green / amber / red). See [CHW Training](CHW_TRAINING.md).
5. **Confirm sync** by triggering a manual sync run while connected and checking
   `last_sync_timestamp` updates in the next site report.

*(Screenshots referenced above are placeholders to be captured on the reference
hardware during the first pilot install and added to `docs/img/`.)*

---

## 4. Daily operational procedures

| When | Action |
|------|--------|
| Clinic open | Power on, confirm the green "ready" state |
| Per patient | Acquire ECG, read the colour tier, act per referral protocol |
| Midday | Glance at the sync icon / run `aortica edge site-report` |
| Clinic close | Ensure a successful sync, then power down cleanly |
| Weekly | Back up `/var/lib/aortica/data`; check storage utilisation |

Generate a daily activity summary for remote monitoring:

```bash
aortica edge site-report --site-id my-clinic --format json
```

This reports the 24-hour inference count, error rate, sync status, storage
utilisation, and last-sync timestamp — the same data exposed by the
`GET /edge/status` endpoint on the local edge server.

---

## 5. Troubleshooting common failure modes

### Power loss

The `aortica-edge` service is configured to restart on boot. After an outage the
device restarts automatically; locally stored results are **preserved** and
re-sync when power and connectivity return. No data is lost by an abrupt power
cut because results are committed to the encrypted store immediately after each
inference.

### SD card corruption

Symptoms: the device fails to boot, or the filesystem mounts read-only.

1. Power down and swap in the **golden recovery card** (Section 2).
2. Re-flash the corrupted card with `create_pi_image_script.sh`.
3. Restore the most recent `/var/lib/aortica/data` backup.
4. Prefer A1/A2-rated industrial SD cards and enable weekly backups to reduce
   recurrence.

### Serial capture timeout

Symptoms: ECG acquisition hangs or reports a timeout.

1. Check the ECG cable and re-seat the connector at both ends.
2. Confirm the capture device has power and is detected
   (`ls /dev/ttyUSB*` or `dmesg | tail`).
3. Retry the recording; ask the patient to remain still to avoid motion
   artefact that can stall capture.
4. If timeouts persist, restart the service: `sudo systemctl restart aortica-edge`.

---

## 6. Localization

The operator-facing strings in this guide and in
[CHW Training](CHW_TRAINING.md) are translatable via JSON locale files in
[`docs/deployment/locales/`](https://github.com/nmillrr/aortica/tree/main/docs/deployment/locales):

- `en.json` — English (complete)
- `fr.json` — French (complete)
- `es.json` — Spanish (template stub)
- `sw.json` — Swahili (template stub)

To add a language, copy `en.json`, translate each value while keeping the keys
unchanged, and drop it in the same directory.
