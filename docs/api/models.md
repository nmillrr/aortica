# Models

Multi-task deep learning engine: backbone, attention, task heads, training, calibration, and uncertainty.

## Backbone

::: aortica.models.backbone.AorticaBackbone

## Cross-Lead Attention

::: aortica.models.attention.CrossLeadAttention

!!! danger "Most outputs are not trained"
    The architecture has 72 outputs; the released model trains **28**:
    AF, AFL, SVT, AVRT, sinus bradycardia/tachycardia, PAC, PVC,
    1st/2nd/3rd-degree AV block, LBBB, RBBB, LAFB, LPFB, WPW, paced rhythm,
    normal sinus rhythm, LVH, RVH, left/right atrial enlargement, STEMI,
    posterior MI, old MI, QTc prolongation, digitalis effect and early
    repolarisation. Every other output below — including VT/VF, Brugada,
    cardiomyopathies, electrolyte disorders and all risk scores — is
    untrained and is withheld from every response. Results for the trained
    outputs: `releases/v0.4.0/RESULTS.md`.

## Task Heads

### Rhythm & Conduction (28 outputs, 18 trained)

::: aortica.models.rhythm_head.RhythmHead

### Structural & Functional (19 outputs, 4 trained)

::: aortica.models.structural_head.StructuralHead

### Ischaemia & Metabolic (19 outputs, 6 trained)

::: aortica.models.ischaemia_head.IschaemiaHead

### Risk Prediction (6 outputs, none trained)

::: aortica.models.risk_head.RiskHead

## Unified Model

::: aortica.models.aortica_model.AorticaModel

::: aortica.models.aortica_model.MultiTaskOutput

## Pre-Trained Model Registry

::: aortica.models.registry.load_pretrained

::: aortica.models.registry.list_available_versions

## Temperature Scaling

::: aortica.models.temperature_scaling.TemperatureScaling

::: aortica.models.temperature_scaling.CalibratedModel

## Conformal Prediction

::: aortica.models.conformal_prediction.ConformalPredictor

::: aortica.models.conformal_prediction.UncertaintyReport
