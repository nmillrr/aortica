# Models

Multi-task deep learning engine: backbone, attention, task heads, training, calibration, and uncertainty.

## Backbone

::: aortica.models.backbone.AorticaBackbone

## Cross-Lead Attention

::: aortica.models.attention.CrossLeadAttention

## Task Heads

### Rhythm & Conduction (22 classes)

::: aortica.models.rhythm_head.RhythmHead

### Structural & Functional (15 classes)

::: aortica.models.structural_head.StructuralHead

### Ischaemia & Metabolic (10 classes)

::: aortica.models.ischaemia_head.IschaemiaHead

### Risk Prediction (3 outputs)

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
