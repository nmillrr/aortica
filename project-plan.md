**AORTICA**

Open-Source AI ECG Platform

**Product Plan & Technical Roadmap**

_Closing Gaps in AI Electrocardiography for Every Clinic - Urban or Rural_

Version 1.0 · March 2026

# **Executive Summary**

Aortica is an open-source AI ECG analysis platform designed to close the most critical gaps in today's clinical ECG landscape: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and near-total exclusion of rural and low-resource healthcare settings.

Built on peer-reviewed research and audited against a detailed technical landscape review, Aortica combines a multi-task deep-learning ECG engine, federated learning infrastructure, clinician-facing explainability tools, edge-optimized inference, and a universal format ingestion layer - all released under a permissive open-source license.

The platform is architected from day one for offline and low-bandwidth environments, requiring no persistent internet connection for core inference. Aortica is not a research notebook or a challenge submission - it is a deployment-ready, modular system that can be adopted by a rural health clinic, a national screening programme, or a device manufacturer alike.

| **Project Name** | Aortica |
| --- | --- |
| **License** | Apache 2.0 (core engine & SDK); CC-BY-4.0 (datasets & docs) |
| **Primary Goal** | Accessible, multi-task, explainable AI ECG analysis for all care settings |
| **Key Differentiators** | Multi-task model · Edge-first deployment · Federated training · XAI with ECG-native language · Universal format support |
| **Initial Target Users** | Rural clinics, primary care physicians, open-source developers, academic researchers, national health programs in LMICs |
| **Regulatory Posture** | Research-grade v1; roadmap to FDA 510(k) / CE-MDR SaMD clearance |

# **1\. Problem Statement**

The AI ECG field presents a paradox: published models report near-perfect accuracy on curated benchmark datasets, yet the gap between laboratory performance and reliable real-world deployment remains vast. The research underpinning Aortica identifies eight interconnected root causes.

## **1.1 Market and Research Gaps**

| **Pain Point** | **Current State** | **Aortica Solution** |
| --- | --- | --- |
| Single-task models | Commercial products address one condition each - AF detection, low EF, or arrhythmia labelling. Clinicians must subscribe to multiple siloed tools. | Aortica's multi-task engine outputs rhythm, structural, ischemia, and metabolic findings from a single forward pass. |
| Poor generalization across devices | Models trained on GE or Philips ECGs degrade significantly when applied to wearables, low-cost 12-lead carts, or single-lead patches - a critical failure for rural settings. | Cross-device domain adaptation layer, calibrated uncertainty estimates, and an explicit signal-quality score per input. |
| Black-box outputs | DL models surface probabilities with no explanation. Clinicians cannot reconcile AI findings with visual ECG interpretation, undermining trust and regulatory compliance. | ECG-native XAI: waveform annotations, named feature attribution (QRS width, ST morphology), and synthetic ECG rendering via a VAE latent space. |
| Inaccessible tooling | Open-source code assumes PhD-level ML expertise, requires Python 3.9 stacks, does not handle heterogeneous formats, and provides no GUI or no-code workflows. | Unified CLI/SDK + optional web UI; universal ECG format reader (WFDB, DICOM, CSV, XML, MAT, HL7 aECG). |
| No rural / edge deployment | All commercially viable products require cloud connectivity. A broken internet connection renders them non-functional. Rural and LMIC clinics are effectively excluded. | Offline-first architecture. Core inference runs on a Raspberry Pi 4 or equivalent ARM device with ≤4 GB RAM. |
| Training data bias | Most public datasets are drawn from single health systems in North America or Western Europe, leading to performance drops in underrepresented demographics. | Federated learning infrastructure, bias auditing per demographic subgroup, and priority partnerships with LMIC data contributors. |
| Fragmented signal quality | Wearable and ambulatory ECGs suffer from motion artifact, muscle noise, and intermittent contact. Many models assume clean hospital-grade signals. | Built-in AI denoising and signal-quality scoring as a pre-processing stage, with configurable reject/flag thresholds. |
| No regulatory scaffolding | Open-source models provide code but no regulatory documentation, test harnesses, performance reports, or post-market surveillance infrastructure. | Embedded validation harness, stratified performance reports (age, sex, ethnicity, device), and a regulatory document template library (IEC 80601-2-86, FDA SaMD). |

# **2\. Vision & Guiding Principles**

Aortica exists on a single conviction: cardiac diagnosis should not depend on geography, device brand, or institutional budget. AI ECG, at its best, is a force multiplier for underserved settings - but only if it is genuinely deployable there.

## **2.1 Core Principles**

| **Principle 1 - Offline First, Cloud Optional** |
| --- |
| Every Aortica feature that can execute locally must execute locally. Cloud integration is additive, never required. A clinic with sporadic 2G connectivity should receive the same quality of inference as a teaching hospital with 1 Gbps fibre. |

| **Principle 2 - Explainability Is Not Optional** |
| --- |
| No prediction ships without a human-readable rationale grounded in named ECG features. Clinicians are partners, not passive recipients of probability scores. XAI is a first-class engineering requirement, not an afterthought. |

| **Principle 3 - Generalization Before Accuracy** |
| --- |
| A model that achieves 99% AUC on a single benchmark but fails in a rural Kenyan clinic has negative clinical value. Aortica optimises for robust cross-device, cross-demographic performance ahead of marginal gains on curated test sets. |

| **Principle 4 - Open by Default, Forkable by Design** |
| --- |
| All model weights, training code, evaluation harnesses, and documentation are released under permissive licences. Any hospital, government, or researcher can fork, adapt, and redistribute without permission or fees. |

| **Principle 5 - Equity as a Technical Requirement** |
| --- |
| Demographic subgroup performance is a gating metric, not a footnote. Models that exhibit statistically significant performance disparities across sex, age, or ethnicity do not pass internal quality gates and are not released. |

# **3\. Target Users & Use Cases**

## **3.1 Primary User Segments**

| **Rural / Primary Care Clinician** | Physician or nurse practitioner in a clinic with a basic 12-lead ECG cart or wearable patch. Needs reliable triage without cardiologist backup. May have intermittent internet. |
| --- | --- |
| **Community Health Worker (CHW)** | Trained lay worker using a smartphone-connected single-lead device or a portable 12-lead in a village or refugee camp. Needs offline operation and simple risk flags, not full clinical reports. |
| **National Screening Programme** | Public health authority running population AF, HF, or ischaemia screening. Needs a deployable pipeline that ingests batch ECG files, outputs structured results, and produces equity-stratified performance audits. |
| **Academic Researcher / ML Engineer** | Wants a reproducible baseline, pre-trained weights, and a benchmark harness to test new architectures against. Values clean APIs, containerised environments, and documented evaluation protocols. |
| **Device Manufacturer** | Wants to embed AI ECG in a patch, wearable, or portable cart without building the model stack from scratch. Needs an SDK with a permissive licence and documented regulatory artefacts. |
| **Hospital / ECG Management System** | Seeks to add multi-task AI to existing ECG infrastructure. Needs FHIR/HL7 integration, worklist prioritisation, and structured write-back - not just a PDF report. |

## **3.2 Rural Healthcare: Special Considerations**

Rural deployment is the most constrained and most impactful use case. Aortica's architecture is stress-tested against the following rural realities:

- No reliable internet: all core inference runs offline; results sync when connectivity is restored.
- Low-cost hardware: inference quantised to INT8 for ARM Cortex-A72 (Raspberry Pi 4), NVIDIA Jetson Nano, or any modern Android device (API 26+).
- Limited training: the mobile interface uses plain-language flags ('Possible irregular heart rhythm - refer to physician') rather than diagnostic jargon.
- Device heterogeneity: rural programmes frequently use low-cost Chinese-manufactured ECG carts alongside donated Western equipment; the universal reader handles both.
- Power constraints: the lightweight model variant consumes <200 mW during inference on ARM hardware.

# **4\. Product Architecture**

Aortica is structured as a layered modular system. Each layer can be adopted independently; the full stack provides end-to-end ECG AI capability from raw signal to clinical report.

## **4.1 Layer Overview**

| **Layer 0 - Signal Ingestion** | Universal ECG format reader: WFDB, DICOM/XML (SCP-ECG), CSV, MAT, HL7 aECG, PhysioNet. Normalises lead order, sampling rate, voltage units, and metadata into a canonical in-memory representation. |
| --- | --- |
| **Layer 1 - Signal Quality & Denoising** | AI-based QRS detection, motion-artifact classification, lead-off detection, and signal-quality scoring (0-100). Flags segments below a configurable quality threshold. Outputs a clean, windowed ECG tensor. |
| **Layer 2 - Multi-Task Inference Engine** | The core 1D residual CNN + cross-attention transformer, producing simultaneous outputs across four task heads: rhythm/conduction, structural/functional, ischaemia/metabolic, and risk prediction. |
| **Layer 3 - Calibration & Uncertainty** | Temperature scaling and conformal prediction wrappers that convert raw logits to well-calibrated probabilities and per-prediction confidence intervals. Flags out-of-distribution inputs. |
| **Layer 4 - XAI Module** | Integrated gradient saliency mapped onto named ECG features (QRS width, PR interval, ST slope, T-wave morphology, axis). VAE latent factor reporter. Case-based retrieval from a de-identified reference library. |
| **Layer 5 - Report & Integration** | Structured clinical report generator (JSON-LD, PDF). HL7 FHIR R4 resource output. DICOM SR write-back. REST API and gRPC service wrapper for EHR / ECG management system integration. |
| **Layer 6 - Federated Learning SDK** | Privacy-preserving training infrastructure using Flower (flwr) framework. Differential privacy via OpenDP. Secure aggregation for multi-site model improvement without raw data sharing. |

## **4.2 Inference Engine: Multi-Task Architecture**

The Aortica model is a shared-backbone multi-task network. A single ECG recording passes through:

- **Encoder backbone:** 12-lead 1D ResNet (adapted from PhysioNet CinC 2021 winning architecture) with skip connections at 64, 128, 256 filter widths. Accepts 2.5s to 10s windows; handles 250-1000 Hz sampling rates via adaptive pooling.
- **Temporal attention module:** Multi-head cross-lead attention (4 heads, 64 dims) capturing inter-lead relationships critical for axis determination, bundle branch differentiation, and ischaemia localisation.
- **Task heads (four):** Separate classification heads for (1) Rhythm & Conduction (22 classes), (2) Structural & Functional (15 classes), (3) Ischaemia & Metabolic (10 classes), (4) Risk Scores (3 continuous outputs: 1-year mortality, HF hospitalisation probability, 12-month AF onset risk).
- **Edge variant:** A knowledge-distilled INT8-quantised model (MobileNet-1D backbone, 2.1M parameters) that runs at <120ms per 10s ECG on ARM Cortex-A72. Full accuracy within 3% of the server model on PTB-XL benchmark.

## **4.3 Task Coverage Comparison**

| **Rhythm & Conduction (22 classes)** | AF, AFL, SVT, AVNRT, AVRT, VT, VF, idioventricular, sinus bradycardia/tachycardia, PAC, PVC, 1st/2nd/3rd degree AV block, LBBB, RBBB, LAFB, LPFB, pre-excitation (WPW), pacemaker rhythm |
| --- | --- |
| **Structural & Functional (15)** | LVH, RVH, LVSD (low EF), HFpEF risk, DCM, HCM, ARVC, cardiac amyloidosis, significant aortic stenosis, significant mitral regurgitation, pulmonary hypertension, LA enlargement, RA enlargement, pericarditis pattern, myocarditis pattern |
| **Ischaemia & Metabolic (10)** | STEMI (per territory), posterior MI, occlusive NSTEMI, old MI, hyperkalaemia, hypokalaemia, hypercalcaemia, hypothyroidism ECG pattern, digitalis effect, QTc prolongation risk |
| **Risk Prediction (3 continuous)** | 1-year all-cause mortality score, 12-month HF hospitalisation probability, 12-month AF onset risk (for patients in sinus rhythm at time of ECG) |

# **5\. Explainability System**

Aortica treats explainability as infrastructure, not decoration. The XAI system maps model decisions onto clinical language that a cardiologist or trained GP can verify against the raw trace.

## **5.1 Named Feature Attribution**

Integrated gradients are computed per-lead and projected onto a set of named ECG morphological features via a pre-trained feature segmentation model:

- P wave: axis, amplitude, duration, morphology (bifid, peaked)
- PR interval: absolute duration, dynamic PR prolongation
- QRS complex: duration, axis, amplitude per lead, notching, slurring
- ST segment: slope (upsloping / flat / downsloping), elevation/depression per territory
- T wave: amplitude, symmetry, inversion territory, peaked morphology
- QT/QTc: Bazett and Fridericia corrected intervals

Each active diagnosis surfaces the top 3 contributing feature attributions with their delta-contribution scores, rendered as colour-coded annotations directly on the ECG waveform in the report.

## **5.2 VAE Latent Factor Reporter**

A variational autoencoder trained on the median beat of each 12-lead recording learns a 24-dimensional interpretable latent space. Each dimension is labelled by correlation with standard ECG measurements. The XAI module reports which latent factors are most activated for a given prediction and renders synthetic ECG reconstructions showing how changing a single factor (e.g., 'factor 7: QRS duration') affects the waveform - giving clinicians an intuitive 'what if' tool.

## **5.3 Case-Based Retrieval**

A de-identified reference library of 50,000 ECGs (from public PhysioNet archives) is indexed by latent space embedding. For any prediction, the system retrieves the 3 most similar historical ECGs with verified diagnoses and outcomes, allowing clinicians to compare the current tracing against phenotypically similar cases.

# **6\. Edge & Rural Deployment**

Edge deployment is the most technically challenging aspect of Aortica and the most important for its mission. The following design decisions are made explicitly to support low-resource settings.

## **6.1 Offline-First Deployment Profiles**

| **Profile A - Raspberry Pi Clinic** | INT8 edge model + signal quality module only. 4GB SD card install. Standalone Python CLI. No network required. CSV/PDF report output. Target: rural sub-Saharan African or South Asian primary care clinic. |
| --- | --- |
| **Profile B - Android App** | ONNX-exported edge model via ONNX Runtime Mobile. Single-lead (AliveCor/KardiaMobile) or 6-lead input. Plain-language output ('Low risk', 'Refer for assessment', 'Urgent referral recommended'). Works fully offline; syncs anonymised audit logs when online. |
| **Profile C - Portable Server** | Full model stack on a Mini-PC (Intel NUC or equivalent) serving a 5-10 device clinic network over local Wi-Fi. Web UI. 12-lead ECG cart integration via DICOM DIMSE or SCP-ECG serial port capture. |
| **Profile D - Cloud/Hybrid** | Full model stack deployed as a Docker container on any cloud provider. REST API + FHIR R4 endpoint. Auto-scaling. Suitable for national programmes, hospital networks, or device vendors. |

## **6.2 Hardware Performance Targets**

| **Raspberry Pi 4 (ARM Cortex-A72, 4GB)** | < 350 ms per 10s 12-lead ECG (edge model) |
| --- | --- |
| **NVIDIA Jetson Nano (4GB)** | < 80 ms per 10s 12-lead ECG (edge model) |
| **Modern Android (Snapdragon 660+)** | < 200 ms per 10s single-lead or 6-lead ECG (ONNX mobile) |
| **Intel Core i5 laptop (CPU only)** | < 150 ms per 10s 12-lead ECG (full model) |
| **Server GPU (NVIDIA A10)** | < 8 ms per 10s 12-lead ECG (full model, batch=1) |

## **6.3 Connectivity & Data Sync**

- All inference results stored locally in a lightweight SQLite database with AES-256 encryption.
- When connectivity is available, opt-in anonymised audit log sync to a central quality-monitoring dashboard.
- Federated model updates pushed over HTTPS when bandwidth exceeds a configurable threshold (default: 1 Mbps for 10 minutes).
- Offline-first sync conflict resolution using vector clocks; no data is lost if two devices process the same ECG independently.

# **7\. Federated Learning & Equity Infrastructure**

The most significant long-term limitation of open ECG AI is dataset bias. Aortica addresses this structurally through federated learning, enabling model improvement across diverse populations without requiring any site to share raw ECG data.

## **7.1 Federated Learning Architecture**

- **Framework:** Flower (flwr) federated learning framework with a pluggable aggregation strategy (FedAvg, FedProx, SCAFFOLD).
- **Privacy:** Differential privacy via OpenDP (ε = 1.0 default, configurable per site). Secure aggregation via CKKS homomorphic encryption for gradient exchange. No raw ECG data ever leaves a participating site.
- **Participation model:** Any institution with ≥500 ECGs and an Aortica deployment can join the federated network. A signed data-use agreement (DUA) template is included in the repository. No commercial relationship required.
- **LMIC priority:** Aortica will actively recruit federated partners in sub-Saharan Africa, South Asia, and Latin America in the first 18 months, offering technical integration support and co-authorship on resulting publications.

## **7.2 Equity Gating Metrics**

Every model release must pass the following equity gates before publication:

- No statistically significant (p &lt; 0.05 after Bonferroni correction) performance difference across sex (male/female/intersex where labelled) for any class with &gt;100 test examples.
- AUC for each class within ±0.04 across age deciles 30-80.
- Documented performance on at least two non-Western site validation sets before any model is tagged as 'v-stable'.
- Public performance card published with every release, including demographic subgroup breakdowns.

# **8\. Integration & Interoperability**

Aortica is designed to fit into existing clinical infrastructure, not replace it. Integration paths are documented and tested for the most common ECG and EHR systems.

## **8.1 Supported Formats & Protocols**

| **ECG Input Formats** | WFDB (.hea/.dat), DICOM ECG (Sup 15), SCP-ECG (.scp), HL7 aECG (FDA XML), CSV, MAT (MATLAB), Cardea JSON, PhysioNet Challenge format |
| --- | --- |
| **EHR Integration** | HL7 FHIR R4 (Observation, DiagnosticReport resources), HL7 v2.x ORU^R01, DICOM SR structured report, REST JSON API |
| **ECG Management Systems** | DICOM DIMSE C-STORE/C-FIND for GE MUSE-style systems; SCP-ECG serial port capture for legacy carts; MLLP listener for HL7 v2 feeds |
| **Report Outputs** | PDF (clinical report), JSON-LD (machine-readable), HL7 FHIR DiagnosticReport, CSV (batch analytics), DICOM SR |
| **Authentication** | OAuth 2.0 / OpenID Connect for cloud deployments; local API key for edge deployments; optional SMART on FHIR launch context |

## **8.2 SDK & Developer Experience**

Aortica ships a Python SDK (pip install aortica) and a REST API. Key developer-facing features:

- One-line inference: aortica.predict(ecg_path) returns a structured result object with all four task heads.
- Fine-tuning API: aortica.finetune(dataset, tasks=\['rhythm', 'structural'\]) wraps HuggingFace Trainer with ECG-specific augmentations.
- Benchmark harness: aortica.benchmark(dataset='ptbxl', split='test') reproduces published results and generates subgroup performance tables.
- Docker images for edge (arm64) and server (amd64/cuda) published to GitHub Container Registry on every release.
- Comprehensive documentation site with clinical background, API reference, deployment guides, and regulatory templates.

# **9\. Development Roadmap**

| **Phase** | **Timeline** | **Key Deliverables** | **Success Metrics** |
| --- | --- | --- | --- |
| **Phase 0 Foundation** | Months 1-3 | Repository structure, CI/CD pipeline, universal ECG reader (all formats), signal quality module, PTB-XL benchmark baseline (rhythm, 3-class structural), public documentation site, Docker images (amd64 + arm64) | PTB-XL rhythm F1 ≥ 0.88; signal quality AUC ≥ 0.93; arm64 inference < 500ms; GitHub stars ≥ 200 |
| **Phase 1 Core Engine** | Months 4-8 | Full 22-class rhythm head; 15-class structural head; calibration layer; integrated gradient XAI with named features; VAE latent reporter; FHIR R4 output; REST API; PyPI package (pip install aortica) | Rhythm macro-F1 ≥ 0.90 on PTB-XL; structural AUC ≥ 0.88 for LVSD; 5 external site validations; 500+ GitHub stars |
| **Phase 2 Edge & Rural** | Months 9-14 | INT8 edge model (knowledge distillation); Android ONNX app (beta); offline sync infrastructure; Raspberry Pi deployment guide and SD card image; first LMIC pilot deployment (target: 2 sites); case-based retrieval system | Edge model AUC within 3% of full model; Android app available on Play Store (beta); 2 rural pilot sites active; offline sync round-trip < 30s on 3G |
| **Phase 3 Federated & Equity** | Months 15-20 | Federated learning SDK (Flower); differential privacy integration; equity gating CI checks; ischaemia & metabolic task head; risk prediction head; first federated model release; public performance cards | 5+ federated sites across ≥3 continents; equity gates passed for all v-stable releases; ischaemia STEMI sensitivity ≥ 90% on held-out external test; risk prediction C-stat ≥ 0.72 |
| **Phase 4 Regulatory & Scale** | Months 21-30 | Regulatory document library (IEC 80601-2-86, FDA SaMD pre-submission templates); prospective validation study at 3 sites; FHIR ECG management system plugin; worklist prioritisation module; national programme deployment support | Pre-submission meeting with FDA; CE technical file drafted; 10,000+ ECGs processed in production; 3 published peer-reviewed validation studies |

# **10\. Open-Source Strategy**

## **10.1 Licensing**

| **Model code & training pipelines** | Apache 2.0 - permissive; commercial use allowed; no copyleft obligation. |
| --- | --- |
| **Pre-trained weights** | Apache 2.0 - freely downloadable, redistributable, and fine-tunable. Model weights trained on public data only for the initial release; federated weights released under the same licence. |
| **Documentation & clinical guides** | CC-BY 4.0 - free to adapt and redistribute with attribution. |
| **Reference datasets (PhysioNet-derived)** | Inherit upstream licences (PhysioNet Restricted / Creative Commons). Aortica does not redistribute raw ECG data; it provides download and preprocessing scripts only. |

## **10.2 Governance**

- Project governed by a Technical Steering Committee (TSC) of 5-7 members, majority not employed by any single organisation.
- RFC process for architectural changes; public roadmap issue tracker on GitHub.
- Code of Conduct (Contributor Covenant 2.1); dedicated clinical safety reviewer role for model releases.
- Security vulnerability disclosure policy (90-day coordinated disclosure with CVE assignment).

## **10.3 Community & Sustainability**

- Annual Aortica Challenge: open competition on a held-out multi-site ECG dataset, modelled on PhysioNet/CinC, to drive model improvement and academic engagement.
- Clinical advisory board: 8-10 cardiologists, emergency physicians, and GPs from diverse geographies providing use-case validation.
- LMIC partner programme: technical integration support and co-authorship for institutions in low- and middle-income countries contributing federated data.
- Sustainability path: hosted cloud service (Aortica Cloud) offered on a cost-recovery basis to organisations that do not wish to self-host, with all profits reinvested in core open-source development.

# **11\. Regulatory Strategy**

Aortica will not launch with regulatory clearance; it will launch with regulatory readiness - the documentation, validation infrastructure, and design decisions that enable clearance. Clinical use prior to clearance is scoped as research or decision support, not standalone diagnosis.

| **Initial release label** | Research use only (RUO) / clinical decision support (non-device in the US per FDA Software Policy). Clinician review required for all outputs. |
| --- | --- |
| **FDA pathway (US)** | 510(k) De Novo for multi-task AI ECG SaMD. Predicate: AliveCor Kardia, Tempus ECG-AF, or Anumana Low EF. Pre-submission meeting targeted for Phase 4 (Month 22). |
| **CE marking (EU)** | Class IIa MDR SaMD. Technical file with IEC 80601-2-86 compliance, TRIPOD-AI/STARD-AI reporting, clinical evidence summary from prospective studies. |
| **Prospective validation** | Multi-site prospective study (3 sites, ≥5,000 ECGs, IRB approved) targeting primary endpoints: STEMI sensitivity, AF detection AUC, LVSD PPV/NPV. Published peer-reviewed. |
| **Post-market surveillance** | Automated performance monitoring against labelled subsets; quarterly public performance reports; voluntary adverse event reporting form integrated into the app. |
| **IEC 80601-2-86 compliance** | Algorithm testing documentation (ATD) template included in the repository. CI pipeline enforces minimum performance targets per device class. |

# **12\. Competitive Differentiation**

Aortica occupies a unique position: it is neither a narrow academic model nor a locked proprietary product. The following table maps Aortica against the existing landscape.

| **Capability** | **PTB-XL / PhysioNet models** | **ExChanGeAI** | **iRhythm / Tempus / Eko** | **AliveCor Kardia 12L** | **AORTICA** |
| --- | --- | --- | --- | --- | --- |
| Multi-task (rhythm + structural + ischaemia + risk) | ❌   | Partial | ❌ (each product = 1 task) | ❌   | **✅** |
| Open weights & training code | ✅   | ✅   | ❌   | ❌   | **✅** |
| Offline / edge deployment | ❌   | ❌   | ❌   | Partial | **✅** |
| ECG-native XAI (named features) | ❌   | Partial | ❌   | ❌   | **✅** |
| Federated learning SDK | ❌   | ❌   | ❌   | ❌   | **✅** |
| Universal format reader | ❌   | Partial | Proprietary | Proprietary | **✅** |
| FHIR / EHR integration | ❌   | ❌   | Proprietary | Limited | **✅** |
| Equity gating metrics | ❌   | ❌   | Undisclosed | Undisclosed | **✅** |
| Regulatory document templates | ❌   | ❌   | FDA cleared | FDA cleared | **✅ (templates)** |
| Rural / LMIC deployment support | ❌   | ❌   | ❌   | Partial | **✅** |

# **13\. Risk Register**

| **Model performance on rare classes** | Mitigation: focal loss training, oversampling, synthetic minority augmentation (SMOTE on latent space). Minimum per-class N=100 test samples before release. |
| --- | --- |
| **Harm from misdiagnosis in rural setting** | Mitigation: conservative thresholds for 'urgent referral' flag; mandatory 'AI output - requires clinical review' watermark; offline mode locks out risk scores without clinician acknowledgement. |
| **Dataset bias perpetuation** | Mitigation: equity gating CI checks; active LMIC partner recruitment; public performance cards per demographic. |
| **Regulatory misuse (CE/FDA before clearance)** | Mitigation: prominent RUO labelling; EULA restricts standalone diagnostic use; regulatory team review of any deployment claiming diagnostic use. |
| **Open-source fork with harmful modifications** | Mitigation: Apache 2.0 does not prevent forking; Community Health Use Policy (non-binding but normative) defines expected use; security advisory board monitors known deployments. |
| **Federated poisoning attack** | Mitigation: gradient clipping, anomaly detection on aggregated updates, per-site performance validation before weight incorporation. |
| **Sustainability / maintenance burden** | Mitigation: modular architecture reduces blast radius of dependency updates; Aortica Cloud revenue funds core maintainers; TSC governance prevents single-organisation control. |

# **14\. Success Metrics**

Aortica will be evaluated against three categories of success metrics, reviewed publicly on a quarterly basis.

## **14.1 Technical Performance**

- PTB-XL rhythm macro-F1 ≥ 0.90 (multi-class, 23 classes)
- LVSD (low EF) AUC ≥ 0.92 on PTB-XL + at least one external validation set
- STEMI sensitivity ≥ 90% at 85% specificity on external emergency-department dataset
- Signal quality AUC ≥ 0.93 on noisy wearable data
- Edge model AUC within 3% of full model on PTB-XL across all tasks
- No statistically significant demographic performance gap (p &lt; 0.05) on any class with N &gt; 100 test examples

## **14.2 Adoption & Reach**

- 2,500+ GitHub stars within 12 months of v1.0 release
- 50+ institutions using Aortica in research or pilot deployment within 18 months
- 5+ active federated learning partners across ≥3 continents within 24 months
- 2 rural or LMIC pilot deployments processing real patient ECGs within 18 months
- 3 peer-reviewed publications citing or using Aortica within 24 months

## **14.3 Clinical & Equity Impact**

- Documentation of at least 1 validated clinical workflow improvement (e.g., reduced time-to-referral, increased AF detection rate) at a pilot site
- Performance cards demonstrating equity-gate compliance published with every v-stable model release
- At least 1 national public health screening programme adopting Aortica within 36 months

# **15\. Conclusion**

The AI ECG landscape in 2026 is simultaneously advanced and deeply fragmented. Clinically impressive models exist, but they are locked behind proprietary systems, require cloud infrastructure, address only a single condition each, and are effectively unavailable to the billions of people whose cardiovascular care is delivered in under-resourced settings.

Aortica is built on the conviction that these are engineering and governance failures, not fundamental limitations. By combining a multi-task inference engine, offline-first architecture, federated learning, ECG-native explainability, and a genuinely open licence, Aortica aims to be the platform that finally bridges the gap between state-of-the-art research and accessible, equitable clinical deployment.

The project does not promise to replace cardiologists or to achieve FDA clearance on day one. It promises a rigorous, honest, and open foundation that any clinic, researcher, or government programme can build on - and that improves continuously as more diverse data and clinical expertise flow into the federated ecosystem.

| **The North Star** |
| --- |
| A 38-year-old woman in a rural clinic in Malawi hands a CHW a smartphone with a single-lead ECG attachment. Within 20 seconds, Aortica - running entirely offline on the phone - flags a high-probability AF signal with a plain-language referral recommendation. The CHW acts. The woman receives anticoagulation therapy. A stroke is prevented.<br><br>That is what open-source AI ECG should mean. |