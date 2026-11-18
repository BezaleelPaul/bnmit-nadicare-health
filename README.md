# SOMYTHRA

**The science of understanding the human body as an interconnected living system.**

> Life is not a collection of parts. It is a system of relationships.

SOMYTHRA is an evolving physiological intelligence and digital-twin research project. It explores how signals from the human body can be interpreted together, in context, to help people and communities better understand patterns in wellbeing, activity, recovery, and environmental strain.

The project continues the technical work originally developed under the name **NadiCare**. The rebrand establishes a broader identity; it does not imply that every capability described here is already implemented or clinically validated.

## Why SOMYTHRA?

SOMYTHRA is a constructed name with an invented internal etymology:

- **SOM** — the living vessel: the human body.
- **YTH** — interdependence: the relationships between biological systems.
- **RA** — pattern intelligence: understanding those relationships as a whole.

Together, **SOMYTHRA** represents the study of the body as an interconnected living system. This is a project-created meaning, not a translation from an established language.

## Mission

Make physiological understanding more accessible, contextual, and useful — from an individual's own patterns to community-level research — while respecting consent, privacy, uncertainty, and human dignity.

## Research and product direction

The project is organised around four connected areas:

| Area | Direction |
| --- | --- |
| **SOMYTHRA Core** | Physiological data ingestion, signal-quality assessment, personal baselines, and privacy-aware data management. |
| **SOMYTHRA Twin** | Exploratory models of an individual's changing physiological patterns over time. |
| **SOMYTHRA Field** | Community-oriented tools for wellbeing awareness, environmental strain, and supportive field workflows. |
| **SOMYTHRA Research** | Signal processing, dataset management, reproducible experiments, and model validation. |

These are product and research directions. Their implementation status must be assessed in the codebase and release notes rather than inferred from the names.

## Guiding principles

- **Evidence before claims:** distinguish measured signals, derived estimates, hypotheses, and synthetic data.
- **Signal quality first:** expose missing, noisy, or uncertain measurements instead of silently treating them as reliable.
- **Personal context matters:** favour transparent personal baselines over unsupported universal thresholds.
- **Privacy by design:** collect only what is needed, explain data use, and make consent meaningful.
- **Offline-aware accessibility:** design for constrained connectivity and affordable hardware where practical.
- **Human-centred use:** community and worker tools must support people, never become punitive surveillance.
- **Responsible boundaries:** this project is for research and wellbeing awareness, not a substitute for professional medical assessment, diagnosis, or treatment.

## Development status

SOMYTHRA is under active development. Features, sensors, models, performance figures, and validation results should only be described as available when they are demonstrated by the implementation and documented tests.

Synthetic or simulated measurements must be clearly labelled and must never be presented as real-world physiological validation.

### Active Milestones: SOMYTHRA Vision R001

The repository currently includes **Milestone R001: Camera-Based Physiological Signal Processing Foundation** located in the [`engine/`](engine/README.md) directory.

Key R001 capabilities:
- **Modular rPPG Pipeline:** Camera/video ingestion $\to$ face detection $\to$ skin segmentation $\to$ temporal RGB extraction $\to$ detrending & filtering $\to$ POS & CHROM pulse extraction $\to$ heart rate estimation & quality gatekeeping.
- **Hardware-Free Verification:** Deterministic optical hemodynamic simulation and synthetic face video generation tools enabling reproducible, automated verification without requiring physical hardware.
- **Quality Gatekeeper:** Frequency-domain SNR and tracking validity checks reject corrupted or unstable signals with explicit reasons, preventing hallucinated or fabricated readings.
- **Privacy by Design:** Local frame processing with zero requirement for storing or transmitting raw facial imagery.
- **Documentation & Tests:** Full test suite and documentation available in [`engine/`](engine/README.md).

## Identity

- **Project identity:** SOMYTHRA
- **Previous working name:** NadiCare
- **Re-foundation / rebrand date:** 18 November 2026
- **Identity serial format:** `SOMYTHRA-181126-001` (proposed convention for the first tracked prototype)

The rebrand date records the identity transition; it does not replace the repository's original creation date or project history.

## Repository

The repository name remains unchanged during this initial identity update. Repository renaming, domain selection, trademark clearance, and public launch are separate decisions.

---

*One organism. Many signals. One interconnected system.*
