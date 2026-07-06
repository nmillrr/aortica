# Pilot Deployment Checklist

Complete this checklist before a site goes live. Every item must be signed off.
Use one copy per site. See the [LMIC Pilot Guide](LMIC_PILOT_GUIDE.md) for the
full procedures behind each item.

**Site:** ____________________  **Date:** ____________  **Engineer:** ____________

---

## 1. Hardware verification

- [ ] Raspberry Pi 4 (4 GB+) boots and passes filesystem expansion
- [ ] SD card is A1/A2-rated, ≥ 32 GB; a sealed **golden recovery card** exists
- [ ] ECG capture device connects and is detected by the OS
- [ ] Power source (mains + battery/UPS) covers a full clinic day
- [ ] Surge protection in place where mains power is unstable
- [ ] Enclosure provides ventilation and dust protection

## 2. Network connectivity test

- [ ] Device reaches the central sync server when connectivity is available
- [ ] `aortica edge site-report` runs and reports storage/status
- [ ] A test result syncs and `last_sync_timestamp` updates
- [ ] Offline behaviour confirmed: a result recorded offline uploads later

## 3. Edge model validation (reuses US-041)

- [ ] Edge model loads and produces output on a known test ECG
- [ ] Edge validation harness passes (AUC within tolerance of the full model)
- [ ] Power consumption validated: sustained draw **< 200 mW** on the RPi4
      profile (`aortica.edge.validate_power_consumption`)
- [ ] Duty-cycled inference confirmed (model loads on-demand, not resident)

## 4. CHW competency sign-off

- [ ] Operator can power the device on and reach the "Ready" state
- [ ] Operator can acquire a good-quality ECG and re-try on poor signal
- [ ] Operator correctly interprets green / amber / red tiers
- [ ] Operator states the correct referral action for each tier
- [ ] Operator can verify sync before ending a shift
- [ ] Operator has the [CHW Training](CHW_TRAINING.md) card in their language

Operator name: ____________________  Signature: ____________________

## 5. Ethics / IRB documentation

- [ ] IRB / ethics approval on file for the pilot site
- [ ] Patient consent process and materials available in local language
- [ ] Data governance reviewed: results are encrypted at rest; no raw data
      leaves the site except approved synced summaries
- [ ] Data protection / privacy obligations for the jurisdiction confirmed
- [ ] Incident and adverse-event reporting pathway documented
- [ ] Site principal investigator / responsible clinician identified

---

## Go / No-Go

- [ ] **All sections above complete** → site approved for pilot activation

Approved by: ____________________  Role: ____________  Date: ____________
