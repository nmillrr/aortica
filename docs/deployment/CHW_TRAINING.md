# Community Health Worker (CHW) Training Guide

This is a plain-language, image-first guide for the person operating an Aortica
edge device at a pilot site. No technical background is assumed. Each step is
designed to be demonstrated once and then followed from the pictures.

> **What this device does:** it takes a heart tracing (ECG) and shows you a
> simple colour — **green**, **amber**, or **red** — telling you whether the
> person needs to be referred. It does **not** replace a clinician's judgment.

*(Each step below has a placeholder for a photo to be added to `docs/img/` from
the reference kit. Keep photos large and uncluttered.)*

---

## Step 1 — Turn the device on

*(Photo: finger holding the power button.)*

- Press and **hold the power button for 3 seconds**.
- Wait for the **green light** and the "Ready" message on the screen.
- If nothing happens, check the power cable and battery.

---

## Step 2 — Take the ECG

*(Photo: electrode placement on the patient.)*

1. Ask the patient to sit or lie down and **stay still**.
2. Attach the electrodes exactly as shown in the picture.
3. Press **Start** and wait for the recording to finish (about 10 seconds).
4. If the screen says the signal is poor, re-attach the electrodes and try again.

---

## Step 3 — Read the result (three colours)

The screen shows **one colour**. This comes from the AI's simplified output.

| Colour | Meaning | What it says |
|:------:|---------|--------------|
| 🟢 **Green** | **Low risk** | No immediate action required. Continue routine care. |
| 🟠 **Amber** | **Refer for assessment** | Schedule a follow-up assessment with a clinician. |
| 🔴 **Red** | **Urgent referral recommended** | Seek immediate medical care. Refer without delay. |

A one- or two-sentence summary of the main finding appears under the colour.

---

## Step 4 — Know when to refer

*(Photo: the referral form / phone.)*

- 🟢 **Green** — no referral needed unless the patient has symptoms; follow your
  usual clinic guidance.
- 🟠 **Amber** — **refer for a scheduled assessment** following your site
  referral protocol.
- 🔴 **Red** — **refer immediately**. Do not wait. Follow the urgent referral
  pathway for your site.

**When in doubt, refer.** The colour is a decision-support aid, not a diagnosis.

---

## Step 5 — Check that results were sent (sync)

*(Photo: the sync icon on screen.)*

- The device stores every result safely, even without internet.
- When a connection is available it **uploads** results automatically.
- Before you finish your shift, check the **sync icon shows a recent successful
  upload**.
- If it does not, move the device to where there is a signal, or tell your
  supervisor. No results are lost — they will send later.

---

## Quick reference card

1. **On** — hold power 3 s → wait for green.
2. **Record** — electrodes on, patient still, press Start.
3. **Read** — green / amber / red.
4. **Refer** — amber = schedule, red = now, green = routine.
5. **Sync** — confirm the upload icon before you leave.

---

## Languages

This guide is available in multiple languages. The wording of the colour tiers
and each step comes from the shared locale files in
[`docs/deployment/locales/`](https://github.com/nmillrr/aortica/tree/main/docs/deployment/locales)
(English and French provided; Spanish and Swahili are template stubs awaiting
translation). Ask your supervisor for the version in your language.
