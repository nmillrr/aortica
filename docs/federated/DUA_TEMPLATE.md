# Data Use Agreement — Aortica Federated Learning Network

> **Template version:** 1.0  
> **Effective date:** _______________  
> **Parties:** This agreement is between the Aortica Federated Learning Network Coordinator ("Coordinator") and the participating institution identified below ("Participant").

---

## 1. Definitions

| Term | Definition |
|------|-----------|
| **Model Updates** | Aggregated gradient or weight delta information transmitted from a Participant's local Aortica client to the FL server during a federated training round. |
| **Local Data** | ECG recordings, patient metadata, and associated labels stored at the Participant's site. Local Data never leaves the Participant's network. |
| **Global Model** | The aggregated model produced by the FL server after combining Model Updates from all Participants. |
| **Privacy Budget** | The differential privacy parameter (ε) governing the maximum information leakage per round, as configured in the Aortica DP wrapper. |

---

## 2. Data Retention and Handling

2.1. **Local Data remains on-site.** At no point during federated training is raw patient data transmitted to the Coordinator, FL server, or any other Participant.

2.2. **Model Updates are ephemeral.** The FL server retains per-round Model Updates only for the duration of aggregation. Once the Global Model for a round is computed, individual Model Updates are deleted.

2.3. **Differential privacy is mandatory.** All Model Updates are protected by differential privacy (default ε = 1.0, δ = 1e-5) via the Aortica `DPWrapper` before transmission. The Participant may configure a stricter privacy budget.

2.4. **Secure aggregation available.** When enabled, Model Updates are encrypted via CKKS homomorphic encryption (`SecureAggregator`) so the FL server aggregates in encrypted space and never observes individual updates in plaintext.

2.5. **Data deletion.** Upon withdrawal (Section 5), the Coordinator will confirm in writing that no Model Updates from the Participant remain on any server infrastructure.

---

## 3. Model Update Usage

3.1. **Permitted use.** Model Updates may be used solely for the purpose of training and improving the Aortica Global Model.

3.2. **No reverse engineering.** No party shall attempt to reconstruct, infer, or identify individual patient records from Model Updates or the Global Model.

3.3. **Global Model distribution.** The resulting Global Model is distributed under the same open-source license as Aortica (Apache 2.0). All Participants receive equal access to the Global Model.

3.4. **Model provenance.** Each Global Model release includes a model card documenting: the number of contributing sites (without identifying them), aggregate dataset demographics, per-task performance, and equity gate results.

---

## 4. Publication Rights

4.1. **Joint publications.** Research publications using federated training results shall acknowledge all participating institutions (with consent) and the Aortica project.

4.2. **Independent publications.** Each Participant retains the right to publish research using their Local Data independently, without requiring Coordinator approval.

4.3. **Embargo period.** The Coordinator will provide at least 30 days' notice before publishing federated training results, allowing Participants to review and request corrections.

4.4. **Attribution.** Publications must cite the Aortica project and the PTB-XL dataset (Wagner et al. 2020, PhysioNet) as the base training data.

---

## 5. Withdrawal Process

5.1. **Voluntary withdrawal.** A Participant may withdraw from the Federated Learning Network at any time by providing written notice to the Coordinator.

5.2. **Effect of withdrawal.** Upon withdrawal:
- The Participant's FL client is disconnected and no further Model Updates are transmitted.
- The Coordinator confirms deletion of any stored Model Updates within 14 business days.
- The Participant retains full access to the latest Global Model at the time of withdrawal.
- Prior contributions to already-aggregated Global Models cannot be removed (as they are irreversibly merged by the aggregation algorithm).

5.3. **Coordinator termination.** The Coordinator may terminate a Participant's access if the Participant materially breaches this agreement, after providing 30 days' written notice and opportunity to cure.

---

## 6. Compliance and Audit

6.1. **Regulatory compliance.** Each Participant is responsible for ensuring their participation complies with applicable local regulations (e.g., HIPAA, GDPR, local IRB/ethics board approval).

6.2. **Audit rights.** The Coordinator may request (no more than once per calendar year) a written attestation that the Participant's deployment configuration meets the minimum security requirements outlined in the Onboarding Guide.

6.3. **Incident reporting.** Both parties shall promptly notify each other of any security incident that may affect Model Updates or the FL infrastructure.

---

## 7. Limitation of Liability

7.1. The Aortica software and Global Model are provided "as-is" without warranty. The Coordinator is not liable for clinical decisions made using the Global Model.

7.2. The Global Model is intended as a clinical decision support tool, not a diagnostic device. All outputs require clinician review.

---

## 8. Signatures

| Role | Name | Institution | Signature | Date |
|------|------|-------------|-----------|------|
| Coordinator Representative | | | | |
| Participant Representative | | | | |
| Participant IT/Security Lead | | | | |

---

*This template is provided as a starting point. Participants should have it reviewed by their legal and compliance teams before execution.*
