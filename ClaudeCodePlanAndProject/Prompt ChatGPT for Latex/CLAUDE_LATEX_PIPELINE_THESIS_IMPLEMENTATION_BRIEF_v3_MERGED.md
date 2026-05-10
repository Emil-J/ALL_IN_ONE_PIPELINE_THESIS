# Claude Code Implementation Brief — LaTeX Thesis Rewrite for GPS-Denied UAV Localisation Pipeline

**Version:** v3 — original full implementation brief plus user addendum. This version preserves the previous brief and adds the latest protected-section, chapter-order, tools/dependencies, citation, and Mermaid-diagram requirements.

**Purpose:** This file is intended to be given to Claude Code inside VS Code. It tells Claude Code how to update the existing LaTeX thesis project using the current pipeline code, recent code-change notes, current README files, live-test evidence, and the Claude Code JSONL conversation log.

**Main instruction:** Do not treat this as a request to write a new thesis from scratch. The task is to update the current LaTeX project so the thesis explains the implemented pipeline accurately, cleanly, and defensibly.

---

## 0. Source-material priority

Use the material in this order of authority:

1. **Current pipeline source code** — source of truth for what the system actually does.
2. **Current LaTeX thesis project** — source of truth for the current document structure and existing writing.
3. **Current README/audit/context files** — source of explanations, risks, diagrams, known issues, and terminology.
4. **Live-test outputs and analysis plots** — source of empirical results, but only where the corresponding CSV/results evidence exists.
5. **Claude Code JSONL conversation log** — background context only. Do not cite it as proof. Do not copy internal conversation wording into the thesis.

The JSONL file is useful to understand how the code evolved, why certain design decisions were made, and which fixes were attempted. However, the thesis must be grounded in the current code and validated outputs, not in chat history.

---

## 1. Goal

Update the LaTeX thesis so it explains the software/simulation pipeline in a formal MSc thesis style.

The thesis should clearly explain:

1. The overall GPS-denied-after-initial-prior localisation problem.
2. The full system architecture before diving into individual components.
3. The offline reference-map and dataset construction flow.
4. The semantic terrain segmentation model and its runtime role.
5. The visual localisation pipeline: query preprocessing, heading rotation, tile search, SuperPoint+LightGlue matching, meta-tile construction, homography, visual quality gates, and look-ahead correction.
6. The EKF/sensor-fusion pipeline: initial geodetic prior, IMU propagation, visual update, adaptive measurement covariance, innovation rejection, and relocalisation recovery.
7. The MSFS 2020/SimConnect live implementation and output files.
8. The experimental results and limitations without overclaiming.
9. The code-level implementation at the correct level of abstraction: module/function responsibility and data flow, not raw code dumps.


---

## 1A. User Addendum — Mandatory structural constraints to merge into this brief

This section is a **mandatory addendum** to the existing brief. It does **not** replace the rest of this file. Claude Code must keep all earlier instructions in this brief and additionally follow the constraints below.

### 1A.1 Protected LaTeX content — do not touch
The following parts of the current LaTeX thesis have already been manually reviewed and fixed by the user. They must not be rewritten, reworded, reformatted, moved destructively, or “improved” without explicit approval:

1. The thesis title.
2. The first three pages / frontmatter pages.
3. Section **4.1 Overview**.
4. Section **4.2 Aerial Imagery and Semantic Mask Data Sources**.
5. Section **4.5 Dataset Generation Workflows**.

Claude may reference these sections, add bridging text before or after them, or propose structural movement if absolutely necessary. However, any plan that modifies the protected text itself must stop and ask for approval first.

### 1A.2 Preferred thesis narrative order
The current LaTeX structure is mostly acceptable, but the technical chapters should be reorganised or bridged so the reader experiences the methodology in this order:

1. **Introduction** - do this last, it should be on top, but this comes last(you need context and understanding of everything).
2. **Literature Review** — revised and better alternative but do this also last it should be on top but last to make(you need context and understanding of everything and references).
3. **System Architecture & Methodology** — first technical overview.
4. **Dataset and Reference Map Construction** — with two clear parts: dataset generation and dataset preprocessing.
5. **Semantic Terrain Segmentation Model Training**.
6. **Simulation Environment** — MSFS2020 / SimConnect / file-mode and live-mode source logic.
7. **Sensor Fusion & EKF**.
8. **Visual Localization Pipeline** — the integration chapter that connects the previous chapters.
9. **MSFS2020 Live Implementation**.
10. **Experimental Evaluation and Results**.
11. **Discussion**.
12. **Conclusion and Future Work**.

Do not blindly renumber everything. First inspect the existing LaTeX structure and propose the minimum edits needed to make this narrative order clear while preserving protected sections.

### 1A.3 Dataset chapter must distinguish generation vs preprocessing
The dataset chapter must explicitly separate two concepts that are currently easy to mix together:
The documentation of what is happening or what was done for the Dataset can be read from the current LaTeX text that I told 
you to not touch, furthermore more details that you can add on the current text or to understand a bit more indepth you
can find in the folder "C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\QGIS"

#### A. Dataset generation
This is the QGIS/geodata side. It should explain:

- the aerial imagery and map-data sources,
- how QGIS was used,
- how semantic masks were generated from geospatial layers,
- how the user selected areas/locations,
- how class colours/layers were defined,
- why the dataset decisions were made,
- how the source imagery and vector layers become aerial/mask tile pairs.

The existing protected sections **4.1**, **4.2**, and **4.5** already contain reviewed material related to this. Do not rewrite them. Build around them and if you wish to change or add on to the existing text, ask for approval via the designated prompt "Yes", "No", "Tell Claude what to do" that Claude code has built in.

#### B. Dataset preprocessing
This is the custom code side. It should explain how the QGIS-generated tile outputs are transformed into runtime-ready artefacts:

- selecting/exporting a reference area from QGIS,
- pointing the preprocessing code(C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Dataset_Preprocessing) to the exported aerial imagery,
- generating a reference map dataset,
- creating the SuperPoint feature database / HDF5 reference database if used in the current code,
- creating semantic prediction tiles from the trained model,
- preparing `aerial/`, `prediction/`, and `reference_features.h5`-type artefacts for runtime localisation,
- validating that the generated artefacts match the TMS tile layout expected by the runtime pipeline.

This preprocessing explanation belongs in the dataset/reference-map chapter, not only in the runtime visual localisation chapter.

### 1A.4 Tools section is required
Add or create a clear **Tools** section/subsection. It should include only tools that matter for reproducing or understanding the implemented workflow.

Possible tools to document if confirmed by the current files/context:

- QGIS for geodata visualisation, layer styling, tile export, and mask generation.
- QMetaTiles or equivalent QGIS tile-export logic if present in the current workflow.
- Osmium / OSM processing tools if relevant to the dataset-generation explanation.
- GDAL/OGR / `ogr2ogr` if used for GeoJSON/GPKG conversion.
- Python scripts/modules built by the user for dataset preprocessing, feature extraction, semantic preprocessing, runtime processing, and analysis. 
- MSFS2020 Simulator and SimConnect for simulation/live input explanation, choice of why, how it works and explanation of code.
- LaTeX/Overleaf/Git/VS Code only if directly relevant to reproducibility or workflow documentation.

Do not write a generic software shopping list. Explain each tool by its actual function in this thesis workflow and if you want to have a mermaid diagram for some of the components, write code for a jpg and then comment and say what you want the diagram to show, then create it and then write in the comment what is the name of the .mmd so i can extract jpg and put it in tex format.

### 1A.5 Dependency/environment requirements are required
Add a section/subsection for the computational environment and dependency requirements. It should cover:

- `.final_Pipeline_venv` / Python environment purpose,
- required Python version if known,
- dependency files such as `requirements.txt`, `environment.yml`, or project-specific dependency notes if present,
- major package groups inferred from the current code only if no dependency file exists,
- GPU/CUDA/PyTorch requirements if the code depends on them,
- OpenCV, NumPy, Pandas, PyTorch, segmentation/model packages, LightGlue/SuperPoint, HDF5 packages, SimConnect packages, and QGIS/GDAL-related tooling only where confirmed.

If dependencies are inferred from imports rather than a lock file, say that clearly. Do not invent exact version numbers unless the files provide them.

### 1A.6 Visual localisation chapter must act as the integration chapter
The visual localisation chapter should not repeat all previous chapters from scratch. It should reference and connect them.

It should explicitly link:

- the dataset/reference-map artefacts from the dataset chapter,
- the trained segmentation model from the semantic chapter,
- the simulation frame/IMU source from the simulation chapter,
- the EKF state and gating logic from the sensor-fusion chapter,
- the runtime pipeline logic from the actual code.

The chapter should explain the operational runtime sequence chronologically:

1. query frame and telemetry arrive,
2. query frame preprocessing and heading rotation,
3. semantic inference / semantic mask generation,
4. tile search / meta-tile generation,
5. SuperPoint + LightGlue matching,
6. homography estimation and visual measurement extraction,
7. semantic confirmation / reliability signal,
8. visual quality and innovation gating,
9. EKF visual update or fallback/coasting,
10. logged outputs and trace artefacts.

### 1A.7 Mermaid diagrams must be regenerated from the current code
The old Mermaid diagrams are outdated and must not be copied blindly.

Claude must generate new diagrams based on the current code and current file structure. The old diagrams may be used only as rough style references.

For each major code-heavy section/subsection/subsubsection, create diagrams that show input/output flow and module responsibility. Diagrams should be created for at least, it is very important for these diagrams to be connected also to text, so it would be wise to first write all of the new stuff in the tex file and then after that context is understood a need is created for a diagram and then the diagram can be made for that code, it is also very important that the diagrams aren't very very large, and if they are very large they should be simplified a lot, because we have many smaller components that create the larger picture all together, so they need to be broken down and then referenced inside the bigger mermaid diagrams that are connecting these components together, think of it as an architect, you can have the bricks and the cement to glue it together but if you don't have the foundation you can't get far, same goes with this, you need to create teh foundation diagrams of all the smaller to larger components and then create the larger diagrams that define subsystems, systems and whole pipelines:

1. System architecture overview.
2. Dataset generation workflow.
3. Dataset preprocessing/reference artefact workflow.
4. Semantic training workflow.
5. Simulation / live-input workflow.
6. Sensor fusion / EKF update workflow.
7. Visual localisation runtime workflow.
8. Output/logging/evaluation workflow.

Diagram requirements:

- Store Mermaid sources in a dedicated diagrams folder if the project already has one, or propose one before creating it.
- Use `.mmd` source files where possible.
- Export to PDF/PNG/SVG only if the existing LaTeX workflow supports it or if the user approves.
- Each diagram must have a clear caption and label plan for LaTeX.
- Every diagram must match the current implementation. If a diagram simplifies something, say so in the caption/text.
- Search/use the JSONL context only to understand earlier diagram intentions, not as technical proof.

### 1A.8 Citation/evidence rule for all newly written text
Every technical statement must be supported in one of these ways:

1. Literature reference.
2. Official documentation reference.
3. Directly observable implementation fact from the current code.
4. Directly observable empirical result from current run outputs.
5. Explicitly labelled engineering judgement / empirical design choice.

Claude must not invent references or fake BibTeX entries.

If a citation is missing, insert a clear placeholder such as:

```latex
% TODO: add citation for QGIS/GDAL/OSM/SimConnect/etc.
```

or add a clearly named BibTeX TODO entry only if the thesis workflow already uses such placeholders.

Preferred wording for unsupported project-specific choices:

```latex
This value was selected empirically during the implementation and validation stage rather than derived as a universal optimum.
```

Do not write unsupported claims such as:

- “This guarantees robustness.”
- “This is the best approach.”
- “This always improves localisation.”

### 1A.9 Implementation mode requirement
Claude Code must first produce a plan, not immediately edit the LaTeX.

The plan must list:

- protected files/sections that will not be touched,
- files Claude proposes to edit,
- new sections/subsections to add,
- diagrams to regenerate,
- citation gaps,
- dependency/environment section location,
- whether the current chapter order needs renaming or only bridging text.

Only after approval should Claude implement.

---

---

## 2. Critical thesis terminology rules

### 2.1 Correct GPS-denied claim

Use this wording:

> The system is **GPS-denied after an initial geodetic prior**.

Do **not** write:

- “GPS-free from startup”
- “never uses GPS”
- “fully GPS-independent from the beginning”

The current pipeline uses an initial latitude/longitude/altitude and heading prior to bootstrap the EKF and search region. In live SimConnect mode, the simulator GPS fields are also logged as ground-truth/evaluation references, but the current corrected runtime code should not feed simulator truth back into the EKF after initialisation.

### 2.2 Correct platform claim

This is a simulation-validated research prototype, not a flight-proven physical UAV system.

Use:

> fixed-wing UAV simulation / simulated fixed-wing aircraft / MSFS 2020 validation platform

Do not imply that the final system has been validated on a physical aircraft unless the user provides real-world flight data.

### 2.3 Correct semantic role

The semantic segmentation model is **not** the primary locator. It is used as:

- a semantic pre-filter for candidate tiles,
- a histogram-based confirmation/reliability signal,
- an auxiliary consistency measure.

Do not claim that semantic segmentation directly estimates precise GPS position. Its current comparison is class-distribution based, not dense spatial semantic registration.

### 2.4 Correct results claim

The old 63-frame CPH result in the existing thesis appears stale compared with the current Odense live tests and changed code. Do not present it as the final headline result unless the user explicitly says to keep it.

If a latest CSV is not present for `live_025`, say that the result numbers must be regenerated or confirmed before final thesis writing.

---

## 3. Uploaded/current project structure found

The LaTeX project currently contains:

```text
LaTeX Thesis/
├── main.tex
├── bibliography.bib
├── Chapters/
│   ├── 01_Introduction.tex
│   ├── 02_LiteratureReview.tex
│   ├── 03_SystemArchitecture.tex
│   ├── 03_Dataset.tex
│   ├── 04_SemanticSegmentation.tex
│   ├── 05_VisualLocalization.tex
│   ├── 06_SensorFusion.tex
│   ├── 07_MSFS.tex
│   ├── 08_Results.tex
│   ├── 09_Discussion.tex
│   └── 10_Conclusion.tex
├── Backmatter/Appendix.tex
├── Frontmatter/
├── Setup/
└── Figures/
```

The `main.tex` currently includes `03_SystemArchitecture.tex` before `03_Dataset.tex`. Keep that order. It is correct: the reader should understand the global pipeline before reading the dataset and component details.

There is also `Chapters/02_Examples.tex`, a long implementation-style document. It is **not included in `main.tex`**. Mine it for useful formulas/explanations only; do not insert it wholesale.

---

## 4. Current code-level source of truth

The current code package contains these key modules:

```text
Pipeline_3_Rev1/
├── runtime/
│   ├── run_pipeline.py
│   └── simconnect_adapter.py
├── src/
│   ├── best_first_search.py
│   ├── ekf_ins.py
│   ├── geometric_matcher.py
│   ├── image_utils.py
│   ├── meta_tile_builder.py
│   ├── particle_filter.py
│   ├── position_estimator.py
│   ├── semantic_confirmer.py
│   ├── semantic_model.py
│   ├── semantic_tile_scorer.py
│   ├── temporal_searcher.py
│   ├── tile_utils.py
│   ├── visual_measurement.py
│   └── wmm_declination.py
└── config/config.py
```

The source-code flow to reflect in the thesis is:

1. `runtime/run_pipeline.py` parses CLI arguments, creates run directories, loads models, initializes EKF, and runs either file replay or SimConnect live mode.
2. `runtime/simconnect_adapter.py` provides input frames and IMU rows from either recorded files or MSFS SimConnect.
3. `src/temporal_searcher.py` owns the per-frame visual localisation pipeline.
4. `src/meta_tile_builder.py` performs two-pass tile search and meta-tile verification.
5. `src/geometric_matcher.py` wraps SuperPoint and LightGlue.
6. `src/semantic_model.py` loads the UNet++/EfficientNet-B3/scSE segmentation model.
7. `src/semantic_tile_scorer.py` and `src/semantic_confirmer.py` perform histogram-intersection semantic scoring.
8. `src/visual_measurement.py` performs heading rotation, dual homography estimation, CShape scoring, and visual measurement extraction.
9. `src/ekf_ins.py` implements the 10-dimensional error-state EKF.
10. `src/tile_utils.py` performs TMS coordinate conversion and tile loading.

---

## 5. Recent code-change state that must be reflected carefully

The current code appears to include these post-audit fixes/changes:

### 5.1 `ver_matches` crash fix

`runtime/run_pipeline.py` now contains `_extract_meta_quality(result)` and uses it in both the file-mode helper and the live SimConnect mode. This prevents the earlier live-mode crash where `ver_matches` was used inside the relocalisation condition without being defined.

Thesis implication: this is an implementation robustness fix, not an algorithmic contribution. It should not be discussed in the main method unless writing a software validation subsection.

### 5.2 Relocalisation recovery logic

`runtime/run_pipeline.py` includes `relocalization_candidate` and `relocalization_applied` logic. It attempts to recover the EKF after consecutive strong visual measurements are rejected by the innovation gate.

Important: uploaded/result evidence showed `relocalization_applied = 0` in the available run CSVs. Therefore, do not claim relocalisation improved the tested results unless a newer run demonstrates it.

Suggested wording:

> The runtime includes a conservative relocalisation recovery mechanism for cases where repeated, geometrically strong visual measurements are rejected by the EKF innovation gate. In the analysed runs, this mechanism remained inactive, so it is treated as a protective mechanism rather than a demonstrated source of accuracy improvement.

### 5.3 Position-estimator Y-axis correction

`src/position_estimator.py` now appears to use the corrected north-up/TMS mapping:

```python
tile_y_frac = (y_max + 1) - ref_px_y / tile_px
tile_y_frac = (tile_y + 1) - ref_px_y / tile_px
```

This should be reflected only if discussing coordinate conversion correctness. Do not over-focus on the bug history in the main thesis.

### 5.4 Trace `pf_center` correction

`src/temporal_searcher.py` now stores both:

```text
pf_center
ekf_center
```

This is a diagnostic/trace correction. It should not be described as an algorithmic change.

### 5.5 RNG centralization

`src/particle_filter.py` now uses one `self.rng = np.random.default_rng()` inside the particle filter object. This centralizes randomness, but it does **not** yet appear to expose a config seed parameter.

Thesis implication: do not claim deterministic replay unless a seed is actually passed/configured. A good future/software-validation note is:

> For fully reproducible replay experiments, the particle filter random generator should be initialized from a fixed seed stored in the run metadata.

### 5.6 Algorithm constants moved to config

`LOOKAHEAD_M`, `R_HIGH`, `R_MED`, `R_COLD_START`, `TURN_ROLL_THRESHOLD_RAD`, and `TURN_R_MULTIPLIER` appear in `config/config.py` and are used through `config.*` in `run_pipeline.py`.

Thesis implication: present these as configurable parameters, not as hidden hard-coded values.

### 5.7 Homography winner score caution

`visual_measurement.py` documentation mentions inlier ratio, but the current scoring code still appears to use absolute inlier count multiplied by CShape and convexity bonus:

```python
score = inliers * CShape * convexity_bonus
```

Do not claim inlier-ratio-based homography selection unless the source code is changed and validated.

---

## 6. Global LaTeX editing rules for Claude Code

1. Work in **Plan Mode first**. Show the exact file-by-file plan before editing.
2. Do not edit frontmatter unless explicitly requested.
3. Do not modify `Setup/`, SDU formatting, bibliography style, title page, or approval pages unless compilation requires it.
4. Preserve the main chapter structure unless there is a strong reason to adjust a section heading.
5. Remove all visible drafting comments such as `\textbf{not sure...}`, `\hl{...}`, and informal notes.
6. Do not paste long code blocks in the thesis body. Use module tables, equations, pseudocode, and short snippets only where academically useful.
7. Use existing citation keys where possible. Do not create fake citations.
8. Do not invent performance numbers. If current result files are insufficient, write placeholders or ask the user.
9. Keep tone formal and academic.
10. Use British/European spelling consistently: localisation, visualisation, behaviour, modelling, etc.
11. Avoid “the drone” when the evidence is MSFS simulation. Prefer “the simulated UAV”, “the fixed-wing platform”, or “the aircraft state”.
12. Compile after editing.

---

## 7. Proposed thesis structure

Keep the current chapter structure:

```text
1 Introduction
2 Literature Review
3 System Architecture and Methodology
4 Dataset and Reference Map Construction
5 Semantic Terrain Segmentation
6 Visual Localisation Pipeline
7 Sensor Fusion and State Estimation
8 MSFS 2020 Live Implementation
9 Experimental Evaluation and Results
10 Discussion
11 Conclusion and Future Work
Appendices
```

Current file names can remain as they are, even though two files begin with `03_`. Do not rename files unless the user approves.

The key structural correction is conceptual, not file-level:

- `03_SystemArchitecture.tex` must explain the whole pipeline end-to-end.
- Component chapters must not be forced to carry the system-level explanation alone.

---

# 8. File-by-file edit plan

## 8.1 `Chapters/01_Introduction.tex`

### Current problems

- Contains visible drafting comments and highlighted uncertainty.
- Problem statement is currently too informal/unclear.
- Some wording overstates real-world UAV validation.
- The initial GPS/geodetic prior must be framed precisely.

### Required changes

Rewrite:

- Motivation
- Problem Statement
- Scope and Delimitations
- Thesis Contributions
- Thesis Structure

Keep or revise Research Questions depending on current wording.

### Required problem-statement wording

Use something close to:

```latex
The problem addressed in this thesis is the estimation of a simulated fixed-wing UAV's geodetic position from camera and inertial measurements after GNSS has become unavailable, assuming that an initial geodetic prior and a geographically referenced map of the operating area are available. The objective is not to construct a map online, but to recover an absolute map-referenced position by matching each query frame against a pre-built TMS reference map and fusing accepted visual measurements with inertial propagation in an error-state Kalman filter.
```

### Required scope bullets

Include these points:

```latex
\begin{itemize}
    \item The system assumes an initial geodetic prior at startup; it is therefore GPS-denied after initialisation, not GPS-free from startup.
    \item The reference map is available before runtime as TMS aerial imagery tiles with known geographic extent.
    \item Reference semantic prediction tiles and SuperPoint feature descriptors are pre-computed offline to reduce runtime load.
    \item Runtime validation is performed in Microsoft Flight Simulator 2020 through SimConnect, not on a physical UAV.
    \item The implementation targets proof-of-concept localisation performance and software integration, not certified real-time flight readiness.
\end{itemize}
```

### Contribution wording

Use contributions such as:

1. Construction of a TMS-based aerial reference-map dataset and semantic mask workflow.
2. Training/integration of a lightweight semantic terrain segmentation model for semantic confirmation.
3. SuperPoint+LightGlue visual matching against map tiles with meta-tile verification.
4. Error-state EKF fusion of visual position updates and inertial measurements.
5. MSFS/SimConnect runtime implementation with traceable per-frame diagnostics.

Do not claim novelty beyond the project scale.

---

## 8.2 `Chapters/02_LiteratureReview.tex`

### Current problems

- Contains `\hl{...}` markers.
- Some sections likely need smoother transitions.
- Literature should support the actual components, not become a generic survey.

### Required changes

Structure the literature review around the pipeline components:

1. GNSS-denied navigation and the need for absolute map correction.
2. Visual-inertial odometry and its drift limitation.
3. Map-based visual localisation / visual place recognition.
4. Local feature matching: SIFT/ORB/SuperPoint/LightGlue/LoFTR.
5. Robust homography estimation: RANSAC/MAGSAC and geometric consistency.
6. Semantic segmentation for aerial terrain understanding.
7. Kalman filtering and error-state inertial navigation.
8. Tile-map infrastructure and TMS/orthophoto references.

Use existing bibliography keys where possible:

```text
kok2017, detone2018, lindenberger2023, sarlin2020, sun2021,
lowe2004, rublee2011, barath2020, fischler1981, ronneberger2015,
zhou2019, tan2019, roy2018, lin2017, salehi2017, doucet2001,
qgis2023, datafordeler_orto_foraar_wmts, bing_maps_aerial,
msfs_bing_data_article, geofabrik_denmark, OpenStreetMap
```

Do not add new references unless required and verified.

---

## 8.3 `Chapters/03_SystemArchitecture.tex`

This chapter is the most important structural chapter.

### Required purpose

It must answer:

> How does the complete system work before individual components are explained?

### Required section structure

Replace or heavily revise with:

```latex
\chapter{System Architecture and Methodology}
\label{chap:system_architecture}

\section{Purpose of the System Architecture}
\section{Offline--Online Pipeline Separation}
\section{Top-Level Data Flow}
\section{Reference Map and Runtime Artefacts}
\section{Runtime Frame Processing Sequence}
\section{Coordinate Frames and Position Representations}
\section{Visual Measurement Acceptance and EKF Fusion}
\section{Software Module Decomposition}
\section{Summary}
```

### Required architecture narrative

The system has two phases:

#### Offline phase

1. QGIS/QMetaTiles exports aerial TMS reference tiles.
2. OSM/vector/geospatial layers are rasterised into semantic masks.
3. Semantic model is trained on aerial-mask pairs.
4. The trained semantic model creates prediction tiles for the reference map.
5. SuperPoint features are extracted for reference tiles and stored in HDF5.

#### Online/runtime phase

1. A camera frame and IMU row are acquired.
2. EKF propagates the state using IMU/barometer/airspeed/magnetometer.
3. Query image is rotated by negative heading for north-up feature matching.
4. Query image is separately resized/padded for semantic inference.
5. Candidate TMS tiles are selected around the current EKF/search state.
6. Semantic pre-filter optionally ranks candidate tiles by histogram intersection.
7. SuperPoint+LightGlue matches the query against candidate tiles.
8. Top candidates form a meta-tile.
9. The meta-tile is verified by a second feature match.
10. A robust homography maps query pixels to reference/meta-tile pixels.
11. A visual position is extracted and look-ahead corrected.
12. The EKF accepts or rejects the visual update using quality and innovation gates.
13. Results and diagnostics are written to CSV/trace outputs.

### Required module table

Add a table like:

```latex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.28\textwidth}p{0.62\textwidth}}
\toprule
\textbf{Module} & \textbf{Runtime role} \\
\midrule
\texttt{runtime/run\_pipeline.py} & Orchestrates file or SimConnect runtime, EKF updates, visual fusion, logging, and output generation. \\
\texttt{src/temporal\_searcher.py} & Executes cold-start and temporal visual localisation for each query frame. \\
\texttt{src/meta\_tile\_builder.py} & Performs semantic pre-filtering, two-pass tile search, meta-tile construction, and verification. \\
\texttt{src/geometric\_matcher.py} & Wraps SuperPoint and LightGlue for feature extraction and matching. \\
\texttt{src/visual\_measurement.py} & Performs heading rotation, homography estimation, CShape scoring, and measurement extraction. \\
\texttt{src/semantic\_model.py} & Loads the UNet++ semantic segmentation model and predicts query class masks. \\
\texttt{src/semantic\_tile\_scorer.py} & Scores candidate reference tiles by semantic class-distribution similarity. \\
\texttt{src/semantic\_confirmer.py} & Computes final semantic histogram confirmation against the meta-tile prediction map. \\
\texttt{src/ekf\_ins.py} & Implements the error-state EKF, inertial propagation, sensor updates, and visual position updates. \\
\bottomrule
\end{tabular}
\caption{Primary software modules and their roles in the runtime localisation pipeline.}
\label{tab:runtime_modules}
\end{table}
```

### Required figure placeholders

If diagrams exist, insert them. If not, create placeholders/comments for the user to add figures later.

Recommended figures:

1. `fig:system_architecture_overview` — offline/online architecture.
2. `fig:runtime_frame_sequence` — one-frame chronological runtime sequence.
3. `fig:artefact_flow` — artifacts: `best.pth`, `prediction/`, `reference_features.h5`, `results.csv`.
4. `fig:software_module_map` — code module decomposition.

---

## 8.4 `Chapters/03_Dataset.tex`

### Current problems

- This chapter should focus on data construction and reference-map artefacts, not runtime algorithm details.
- It should distinguish training dataset, reference map, prediction map, and feature store.

### Required section structure

Use:

```latex
\section{Overview}
\section{Aerial Imagery and Vector Data Sources}
\section{TMS Tile Grid and Geographic Extent}
\section{Semantic Mask Generation in QGIS}
\section{Aerial--Mask Pair Dataset for Training}
\section{Reference Map Preprocessing for Runtime}
\section{Precomputed Semantic Prediction Tiles}
\section{Precomputed SuperPoint Feature Store}
\section{Dataset Statistics and Class Distribution}
\section{Limitations of the Dataset Construction}
```

### Required content

Explain four different datasets/artefacts:

1. **Training aerial-mask pairs** — used to train the semantic model.
2. **Runtime aerial reference tiles** — used by `TileLoader.load_aerial()`.
3. **Runtime semantic prediction tiles** — generated offline and used by semantic scoring/confirmation.
4. **Runtime SuperPoint HDF5 feature store** — speeds up tile matching.

### Required terminology

Use “aerial imagery” / “orthophoto reference imagery” for GeoDanmark/Ortofoto DK.

For Bing, use “Bing aerial/satellite map imagery” carefully. If discussing MSFS, state that Bing imagery was chosen because MSFS 2020 uses Bing Maps data as part of the simulated world-generation pipeline, reducing the domain gap between simulator view and reference map.

---

## 8.5 `Chapters/04_SemanticSegmentation.tex`

### Current purpose

This chapter should cover training, architecture, loss, imbalance handling, evaluation, and runtime role.

### Required corrections

Do not imply semantic segmentation performs spatial registration. The current semantic scoring is histogram-intersection based.

Explain:

- UNet++ with EfficientNet-B3 encoder and scSE attention.
- Six terrain classes.
- Composite loss: focal + dice + tversky.
- Class imbalance handling.
- Semantic model output: `512 x 512` class-index mask.
- Runtime query semantic mask is produced from the original/resized query frame, while SP+LG matching uses a heading-rotated image.
- Current semantic confirmation is histogram-based and therefore mostly orientation-invariant.

### Required caution about semantic alignment

Add a limitation/future-work paragraph:

```latex
In the current implementation, semantic confirmation is based on class-distribution similarity rather than dense spatial alignment. This choice makes the score less sensitive to heading rotation, but also limits its ability to reject geometrically plausible matches in repeated terrain. A future spatial semantic consistency check would require rotating the predicted query semantic mask using nearest-neighbour interpolation and comparing it against the north-aligned reference prediction map within the homography-supported region.
```

### Do not overclaim semantic confidence

Do not write that semantic confidence reliably detects all wrong matches. Existing run evidence showed bad and good visual cases can have similar semantic confidence values.

---

## 8.6 `Chapters/05_VisualLocalization.tex`

This chapter must match the current runtime code.

### Required section structure

Use:

```latex
\section{Pipeline Overview}
\section{Frame Preprocessing and Heading Rotation}
\section{Cold-Start Tile Search}
\section{Temporal Tile Search and Particle-Guided Radius}
\section{Semantic Candidate Pre-Filtering}
\section{Meta-Tile Construction and Verification}
\section{SuperPoint and LightGlue Matching}
\section{Dual Homography Estimation}
\section{Shape Confidence and Visual Quality Gate}
\section{Visual Measurement Extraction}
\section{Camera Look-Ahead Correction}
\section{Failure Modes}
```

### Required technical facts

1. Query frame is rotated by `-heading` before feature matching.
2. The rotated image is resized/cropped for matching.
3. Semantic inference uses separately resized/padded input.
4. Frame 0 uses `BestFirstSearcher` cold start.
5. Frames N use `TemporalSearcher._process_frame_N()` with particle prediction and meta-tile search.
6. Candidate tiles are selected via `find_tiles_within_radius()` around the EKF/search center.
7. Semantic pre-filter keeps top candidate tiles by histogram intersection when enabled.
8. Meta-tile verification records `verification_matches` and `meta_tile_verified`.
9. Homography branch selection currently uses inlier count, CShape, and convexity bonus; do not claim inlier-ratio selection unless implemented.
10. Visual quality uses CShape/inlier thresholds.
11. Innovation gate rejects visually plausible but EKF-inconsistent measurements.
12. Look-ahead correction uses configurable `LOOKAHEAD_M = 110 m` scaled by `cos(bank)`.

### Suggested equation for look-ahead correction

Use:

```latex
\begin{align}
    d_N &= -L\cos(\psi), \\
    d_E &= -L\sin(\psi),
\end{align}
```

Then mention the implementation scales the effective distance with bank angle:

```latex
\begin{equation}
    L_{\mathrm{eff}} = L \cos(\phi),
\end{equation}
```

where $\psi$ is heading and $\phi$ is bank angle.

Be explicit that this is empirical and camera-configuration dependent.

### Required limitation

Add:

> During banked turns, the visible ground footprint can be displaced from the aircraft nadir point. The current correction is a first-order empirical correction, not a full camera-ground intersection model. This can allow strong image-to-map matches to produce biased aircraft-position updates.

---

## 8.7 `Chapters/06_SensorFusion.tex`

### Required content

This chapter should explain:

1. Nominal state and error state.
2. 10D error state: orientation error, gyro bias error, wind NE error, horizontal position NE error.
3. IMU propagation.
4. Accelerometer/magnetometer/barometer/airspeed updates.
5. Visual position update.
6. Adaptive measurement covariance.
7. Innovation gate.
8. Relocalisation recovery logic as a protective mechanism.

### Required GPS-prior clarification

Add a subsection:

```latex
\subsection{Initial Geodetic Prior and GPS-Denied Operation}
```

Text:

```latex
The EKF is initialised from a single geodetic prior consisting of latitude, longitude, altitude and heading. This prior defines the local navigation frame and constrains the initial visual search region. After this initialisation step, the estimator is designed to propagate using inertial and auxiliary measurements and to correct its horizontal position using accepted visual measurements. Simulator latitude and longitude are retained in the output files for evaluation, but are not part of the GPS-denied estimator update in the corrected runtime path.
```

### Required relocalisation wording

Use conservative wording:

```latex
The runtime also includes a relocalisation recovery mechanism for the case where several consecutive visually strong measurements are rejected by the innovation gate. A frame is considered a recovery candidate only if it satisfies stricter CShape, inlier, meta-tile verification, and verification-match thresholds. If the candidate sequence is coherent over multiple frames, the horizontal position covariance is inflated and a recovery update is applied. In the available validation runs this mechanism should be reported only if the `relocalization_applied` field is non-zero.
```

Do not claim it improved results unless validated.

---

## 8.8 `Chapters/07_MSFS.tex`

### Required content

Explain:

1. MSFS 2020 as validation platform.
2. SimConnect live source.
3. File replay source.
4. Frame/IMU alignment limitations.
5. Output files.
6. Pipeline trace folder.
7. PX4-compatible GPS_INPUT CSV.
8. Runtime latency.

### Required caveat

Mention that live mode is not fully deterministic because frames, SimConnect rows, GPU inference, and runtime scheduling are asynchronous.

If discussing RNG, write:

> The particle filter uses stochastic sampling to represent uncertainty. The current implementation centralises the random generator inside the particle filter, but deterministic replay requires exposing and logging a fixed seed.

---

## 8.9 `Chapters/08_Results.tex`

### Current problem

The current chapter likely contains older results and may no longer reflect the current pipeline state.

### Required instructions

Before editing the numeric results, locate the latest validated `results.csv` and `run_meta.json`.

If only the uploaded 131-frame result is available, use it as a clearly labelled run, e.g.:

```text
live_023_Odense_1ft or uploaded 131-frame Odense run
```

Do not mix old CPH 63-frame results with newer Odense results without separating them.

### Required table fields

Include a table with:

- run ID,
- source mode,
- number of frames,
- gate pass count/rate,
- mean error,
- median error,
- max error,
- mean/median inference latency,
- relocalisation applied count,
- notes.

If `live_025` only contains plots and not CSV, do not invent numbers.

### Required analysis subsections

Use:

```latex
\section{Experimental Setup}
\section{Run Summary}
\section{Visual Gate and EKF Update Behaviour}
\section{Trajectory Accuracy}
\section{Runtime and Latency}
\section{Turn and Bank-Angle Effects}
\section{Semantic Confirmation Behaviour}
\section{Failure Case Analysis}
\section{Result Validity and Limitations}
```

### Required warning about latency

If using the uploaded live run values, explain that inference latency is a navigation limitation. A visual estimate produced 2--3 seconds after frame capture may be geometrically correct for the image time but delayed relative to current aircraft state.

Do not bury this in future work only; it affects interpretation of live operation.

---

## 8.10 `Chapters/09_Discussion.tex`

### Required themes

Discuss:

1. GPS-denied after initial prior vs GPS-free from startup.
2. Why SuperPoint+LightGlue was chosen.
3. Why semantic segmentation is useful but insufficient as a primary locator.
4. EKF temporal continuity and risk of bad visual updates.
5. Banking/oblique view limitations.
6. Initial geodetic prior dependence.
7. Runtime latency and embedded deployment challenges.
8. Simulation-to-real gap.
9. Reference-map dependency.
10. Software robustness issues and reproducibility.

### Required blunt but academic conclusion

The thesis should not pretend the solution is flight-ready. It should say:

> The implemented system demonstrates the feasibility of map-referenced visual-inertial localisation in a simulated GPS-denied setting, but its current form remains a research prototype limited by latency, camera-model simplifications, reference-map dependency, and simulation-only validation.

---

## 8.11 `Chapters/10_Conclusion.tex`

### Required correction

Do not use old claims like “all objectives achieved” if the latest run has unresolved limitations.

Conclusion should state:

1. What was built.
2. What worked.
3. What remains weak.
4. What must be done before real deployment.

### Future work list

Use:

1. Full camera calibration and roll/pitch-aware ground-footprint correction.
2. Latency compensation with delayed-measurement EKF update or timestamp-corrected propagation.
3. Spatial semantic consistency using rotated predicted query masks and nearest-neighbour interpolation.
4. Deterministic replay via configurable particle-filter seed.
5. Stronger homography diagnostics and scoring after instrumentation.
6. Embedded deployment on Jetson/Orin or equivalent.
7. Physical UAV validation.
8. Multi-scale tile search and altitude adaptation.

---

## 8.12 `Backmatter/Appendix.tex`

Use appendix for details that are too heavy for main chapters:

1. Error-state EKF equations.
2. MSFS axis mapping.
3. Runtime output column definitions.
4. Module-level code map.
5. Parameter table.
6. Additional run plots.
7. Class distribution tables.

If adding code snippets, keep them short.

---

# 9. Required LaTeX-ready blocks

## 9.1 System architecture opening paragraph

```latex
The localisation system is implemented as an offline--online pipeline. The offline stage constructs the geographically referenced artefacts required at runtime: aerial TMS reference tiles, semantic prediction tiles, and precomputed SuperPoint descriptors. The online stage consumes a camera frame and an aligned inertial measurement row, propagates the error-state Kalman filter, performs visual localisation against the reference map, verifies the visual measurement using geometric and semantic criteria, and fuses accepted visual positions into the EKF. This separation reduces runtime computation while keeping the reference-map construction reproducible.
```

## 9.2 Runtime sequence paragraph

```latex
For each frame, the runtime first propagates the EKF using the newest inertial and auxiliary measurements. The current EKF state provides the heading, position prior, velocity and uncertainty used by the visual search stage. The query frame then follows two preprocessing paths. The feature-matching path rotates the image by the negative EKF heading to align it approximately with the north-up reference map before SuperPoint feature extraction. The semantic path resizes and pads the original query frame to the network input size and predicts a six-class terrain mask. Candidate TMS tiles are selected around the current search centre, optionally pre-filtered by semantic histogram similarity, matched using SuperPoint and LightGlue, and assembled into a meta-tile. A robust homography maps the query image into the reference meta-tile, from which a geodetic visual measurement is extracted and corrected for camera look-ahead before EKF fusion.
```

## 9.3 Semantic role paragraph

```latex
The semantic model is used as a reliability aid rather than as a standalone position estimator. Its output provides class-distribution information that can reject or down-rank visually implausible candidate tiles, but the histogram-intersection score does not encode metric position within a tile. The final position measurement therefore remains geometric, derived from feature correspondences and homography, while semantic confirmation contributes an auxiliary confidence signal.
```

## 9.4 GPS-denied qualification paragraph

```latex
The system should be interpreted as GPS-denied after initialisation. A single geodetic prior is required to initialise the EKF and to restrict the first visual search to a feasible region of the reference map. After this initial prior, the estimator is designed to rely on inertial propagation and accepted visual map matches. Ground-truth latitude and longitude from MSFS are logged for evaluation, but should not be treated as runtime localisation inputs.
```

## 9.5 Limitation paragraph for banked turns

```latex
A limitation of the current visual measurement model is that the homography estimates the ground area visible to the camera, whereas the EKF requires the aircraft position. During banked flight the visible ground footprint can be laterally displaced from the aircraft nadir point, so a visually strong image-to-map match may still produce a biased aircraft-position update. The implemented look-ahead correction compensates for the dominant forward camera offset, but it is not a full attitude-aware camera-ground intersection model.
```

---

# 10. Figures and tables to insert

## 10.1 Required figures

Create or insert these figures if available:

| Figure | Target chapter | Purpose |
|---|---|---|
| System architecture overview | Chapter 3 | Offline/online pipeline split |
| Runtime frame-processing sequence | Chapter 3 or 5 | Chronological per-frame processing |
| Dataset generation workflow | Chapter 4 | QGIS + masks + TMS export |
| Semantic model architecture | Chapter 5 | UNet++/EffNet-B3/scSE overview |
| Meta-tile search diagram | Chapter 6 | first-pass, second-pass, meta-tile verification |
| EKF fusion diagram | Chapter 7 | IMU predict + visual update |
| Trajectory/error plots | Results | empirical validation |
| Pipeline trace examples | Results or Discussion | good/bad frame examples |

## 10.2 Required tables

Add tables for:

1. Software modules and roles.
2. Offline artefacts and consumers.
3. Runtime output columns.
4. Main configurable parameters.
5. Run-summary metrics.
6. Known limitations and mitigation/future fix.

---

# 11. Parameters table to include

Add a table in System Architecture, Visual Localisation, Sensor Fusion, or Appendix:

```latex
\begin{table}[H]
\centering
\small
\begin{tabular}{p{0.32\textwidth}p{0.20\textwidth}p{0.38\textwidth}}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Role} \\
\midrule
\texttt{LOOKAHEAD\_M} & 110 m & Empirical forward camera-offset correction. \\
\texttt{R\_HIGH} & $30^2$ m$^2$ & EKF visual-measurement variance for high-quality matches. \\
\texttt{R\_MED} & $60^2$ m$^2$ & EKF visual-measurement variance for normal accepted matches. \\
\texttt{R\_COLD\_START} & $100^2$ m$^2$ & Reduced trust for cold-start visual measurement. \\
\texttt{TURN\_ROLL\_THRESHOLD\_RAD} & 0.35 rad & Bank threshold for inflating visual-measurement covariance. \\
\texttt{TURN\_R\_MULTIPLIER} & 2.0 & Measurement-noise inflation during steep bank. \\
\bottomrule
\end{tabular}
\caption{Main configurable visual-fusion parameters used by the runtime EKF update.}
\label{tab:visual_fusion_parameters}
\end{table}
```

Verify values against `config/config.py` before insertion.

---

# 12. What not to do

Do not:

1. Rewrite the thesis into a software manual.
2. Paste hundreds of lines of code.
3. Claim physical UAV validation.
4. Claim pure GPS-free operation from startup.
5. Claim semantic segmentation directly gives sub-tile GPS position.
6. Claim relocalisation improved the run unless `relocalization_applied > 0` is shown in the relevant CSV.
7. Claim deterministic replay unless a fixed RNG seed is implemented and logged.
8. Claim homography winner uses inlier ratio unless the code actually does it.
9. Hide latency limitations.
10. Hide simulation-to-real limitations.
11. Keep any visible draft comments or `\hl{}` markers.

---

# 13. Validation checklist for Claude Code after LaTeX edits

Run from the LaTeX project root:

```powershell
latexmk -pdf main.tex
```

or the project’s existing build command.

Then verify:

```powershell
Select-String -Path Chapters\*.tex -Pattern "\\hl\{|not sure|i think|wtf|bro|TODO|FIXME|textbf\{.*not"
```

There should be no informal drafting comments left.

Also check:

```powershell
Select-String -Path Chapters\*.tex -Pattern "GPS-free|never uses GPS|fully GPS independent|real drone|flight-ready"
```

Any occurrence must be reviewed.

Check references/labels:

```powershell
Select-String -Path main.log -Pattern "undefined|Citation|Reference|Overfull"
```

Warnings are allowed only if understood and reported.

Final Claude Code report must include:

1. Files changed.
2. Sections changed.
3. Build command run.
4. Compilation status.
5. Remaining warnings.
6. Sections requiring user review.
7. Result numbers that still need confirmation.

---

# 14. Implementation order

Claude Code should implement in this order:

1. Scan current LaTeX and list existing chapter/section structure.
2. Scan current source files and confirm the latest code-state facts listed above.
3. Show a plan before editing.
4. Clean obvious drafting comments in `01_Introduction.tex` and `02_LiteratureReview.tex`.
5. Rewrite/strengthen `03_SystemArchitecture.tex` as the global architecture chapter.
6. Update dataset chapter to separate training/reference/runtime artefacts.
7. Update semantic chapter to accurately describe histogram-based runtime role.
8. Update visual localisation chapter to match current runtime sequence.
9. Update sensor fusion chapter with GPS-prior, innovation gate, and relocalisation recovery caveats.
10. Update MSFS chapter and results chapter only after verifying which run outputs are current.
11. Update discussion/conclusion to remove overclaims.
12. Compile and report.

---

# 15. Questions Claude Code must ask before implementation if uncertain

Ask the user if any of these are unclear:

1. Which result run is final for the thesis: old CPH 63-frame run, live_023 Odense, live_024, live_025, or a future rerun?
2. Should `Frontmatter/Abstract.tex` be edited now or left untouched?
3. Should `Chapters/02_Examples.tex` remain unused, be deleted, or be converted into appendix material?
4. Which figures are already approved for inclusion?
5. Should diagrams be generated as Mermaid, exported PNG/PDF, or LaTeX/TikZ?
6. Are manually revised sections off-limits?

If these are not answered, proceed only with the clearly safe chapters and mark uncertain areas for review.

---

# 16. Final expected output from Claude Code

Claude Code should produce:

1. A revised LaTeX thesis that compiles.
2. A concise change log.
3. A list of remaining thesis weaknesses.
4. A list of result numbers that still need validation.
5. A list of figures/tables that still need to be exported or inserted.

The goal is not cosmetic polishing. The goal is to make the thesis technically honest, readable, and aligned with the actual current pipeline.
