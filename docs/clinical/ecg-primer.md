# ECG Primer

A brief introduction to electrocardiography for developers working with Aortica.

## What is an ECG?

An electrocardiogram (ECG/EKG) records the electrical activity of the heart over time using electrodes placed on the body surface. A standard 12-lead ECG provides 12 different "views" of the heart's electrical activity.

## The 12 Leads

| Lead | Type | View |
|------|------|------|
| I, II, III | Limb leads | Frontal plane (bipolar) |
| aVR, aVL, aVF | Augmented leads | Frontal plane (unipolar) |
| V1–V6 | Precordial leads | Horizontal plane |

## The ECG Waveform

Each cardiac cycle produces the **PQRST complex**:

```
     R
    /|\
   / | \
  /  |  \         T
 /   |   \      /   \
P    |    \    /     \
     |     \  /       \
     |      \/         \
     Q      S           U
     
|---PR--|--QRS--|--ST--|--QT------------|
```

| Segment | Duration | Represents |
|---------|----------|------------|
| **P wave** | 80–120 ms | Atrial depolarization |
| **PR interval** | 120–200 ms | AV conduction time |
| **QRS complex** | 60–100 ms | Ventricular depolarization |
| **ST segment** | 80–120 ms | Early ventricular repolarization |
| **T wave** | ~160 ms | Ventricular repolarization |
| **QT interval** | 350–440 ms | Total ventricular activity |

## Key Parameters

- **Heart rate**: 60–100 bpm (normal sinus rhythm)
- **Rhythm**: Regular vs irregular
- **Axis**: Normal −30° to +90°
- **Sample rate**: Typically 250–500 Hz (Aortica normalizes to 500 Hz)
- **Amplitude**: Measured in millivolts (mV); standard calibration is 10 mm/mV

## Signal Quality

ECG signals are susceptible to several types of noise:

- **Baseline wander** — low-frequency drift from breathing or movement
- **Powerline interference** — 50/60 Hz from mains power
- **Muscle artifact** — high-frequency noise from skeletal muscle
- **Lead-off** — flatline when an electrode loses contact

Aortica's signal processing pipeline handles all of these automatically.

## Supported Formats

| Format | Extension | Source |
|--------|-----------|--------|
| WFDB | `.hea` + `.dat` | PhysioNet datasets |
| DICOM | `.dcm` | Clinical PACS systems |
| SCP-ECG | `.scp` | European ECG carts |
| HL7 aECG | `.xml` | FDA submissions |
| CSV | `.csv` | Research exports |
| MATLAB | `.mat` | Research tools |
| PDF/Image | `.pdf`, `.png`, `.jpg` | Scanned paper ECGs |
