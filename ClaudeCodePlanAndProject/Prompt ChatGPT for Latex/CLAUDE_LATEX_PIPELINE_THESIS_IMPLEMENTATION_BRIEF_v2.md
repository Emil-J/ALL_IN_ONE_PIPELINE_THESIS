# Claude Code Implementation Brief — LaTeX Thesis Update (Revised Constraints)

## Purpose
This brief tells Claude Code how to update the current LaTeX thesis workspace **without damaging the reviewed parts**, while restructuring and extending the thesis so that the pipeline, codebase, diagrams, tools, dependencies, simulation setup, sensor fusion, and visual localisation methodology are documented in a rigorous and thesis-appropriate way.

The goal is **not** to rewrite everything blindly. The goal is to:
1. preserve the protected parts,
2. keep the existing LaTeX structure where reasonable,
3. reorganise the technical narrative into a clearer methodology flow,
4. add missing sections/subsections,
5. regenerate Mermaid diagrams so they match the current code,
6. ensure claims are backed either by literature, official documentation, or explicitly marked empirical engineering choices.

---

## Source hierarchy Claude must follow
Use the inputs with the following priority:

### Highest priority — current thesis workspace and current code
- **LaTeX Thesis_current.zip** = source of truth for the current thesis files and structure.
- **pipeline code current.zip** = source of truth for the current implementation.
- **code changes made today.zip** = source of truth for the latest modifications made today.
- **live_tests_today_025_is_latest_code.zip** = source of truth for the latest validation artefacts and runtime evidence.

### Secondary priority — support material
- **some readme files from code changes and plans today.zip** = implementation notes, planning notes, and explanations.
- **mermaid diagrams.zip** = old diagrams for reference only; they are likely outdated and should **not** be trusted blindly.
- **3128fa8b-7f3f-4f65-9d61-1881f0386fc8.jsonl** = historical conversational/project context. Use this only as supporting context, not as sole evidence.

### Rule
If the JSONL context conflicts with the current code or current LaTeX workspace, **trust the current code/workspace**.

---

## Protected / do-not-touch content
The following must be preserved unless I explicitly approve changes:

1. **The title**
2. **The first three pages**
3. **Section 4.1 Overview**
4. **Section 4.2 Aerial Imagery and Semantic Mask Data Sources**
5. **Section 4.5 Dataset Generation Workflows**

### Important instruction
Claude Code may:
- reference these protected sections,
- add linking text around them,
- adjust surrounding structure if needed,
- move where later sections appear in the global chapter flow if necessary,

but must **not rewrite or substantially modify the protected content itself**.

If a structural change requires touching those sections, Claude must stop and propose the exact minimum change first.

---

## Required thesis-level content changes
The thesis should keep the current overall structure where possible, but the technical flow should follow this order:

1. **System Architecture & Methodology**
2. **Dataset Section**
   - Dataset generation
   - Dataset preprocessing
3. **Semantic Terrain Segmentation Model Training**
4. **Simulation Environment**
5. **Sensor Fusion & EKF**
6. **Visual Localization Pipeline**
7. **MSFS2020 Live Implementation**
8. **Experimental Evaluation / Results / Discussion**
9. **Conclusion and Future Work**

This is the preferred narrative order even if the exact chapter numbers need to be adapted to the existing LaTeX project.

---

## Detailed structure requirements

### 1. System Architecture & Methodology goes first
This should appear before the deep dataset and model sections, and should act as the reader’s roadmap.

It should explain at a high level:
- overall system goal,
- main subsystems,
- offline vs online workflow,
- role of QGIS dataset preparation,
- role of semantic segmentation,
- role of feature matching and homography,
- role of EKF/sensor fusion,
- role of simulation/live testing.

This section should orient the reader before technical depth begins.

---

### 2. Dataset section must be split clearly into two subsections
The dataset section needs to contain **two clearly separated subsections**:

#### 2.1 Dataset generation
This subsection should cover:
- use of **QGIS**,
- aerial imagery sources,
- semantic mask sources,
- why those data sources were selected,
- map/layer decisions,
- structure of the dataset,
- logic behind area selection and tile preparation,
- how masks and aerial imagery relate,
- decisions made for dataset construction.

This is conceptually close to what exists now and should connect well to the protected sections.

#### 2.2 Dataset preprocessing
This subsection must explain the **custom preprocessing pipeline code**.
It should describe that an area is selected in QGIS and then the custom dataset preprocessing pipeline:
- processes aerial tiles exported/generated from QGIS,
- builds a **reference map dataset**,
- creates a **reference database**,
- produces **semantic prediction tiles / semantic map outputs**,
- prepares artefacts needed by the runtime visual localisation pipeline.

This subsection should be clearly separated from raw data-source discussion. It is about converting source data into runtime-usable assets.

---

### 3. Semantic Terrain Segmentation Model Training
This section should document:
- training purpose,
- model role within the overall localisation system,
- architecture used,
- input/output definition,
- dataset split logic,
- augmentations/preprocessing,
- loss/metrics,
- training strategy,
- class design and label interpretation,
- validation metrics,
- final role of the trained model in the live/runtime pipeline.

Any claim about architecture or training choice should be backed either by:
- a literature citation,
- official framework documentation,
- or explicit empirical justification.

If a choice is empirical, the wording should say that it was selected based on practical experimentation / project validation, not presented as universal truth.

---

### 4. Simulation Environment
A dedicated section is needed for the simulation setup and relevant code.
This should include:
- simulation environment purpose,
- what MSFS2020 contributes,
- what data it provides,
- how frames and telemetry are captured,
- the role of the simulation environment in development and validation,
- relevant code modules that support file mode or live mode,
- limitations of the simulation environment.

This section should make clear what is simulated, what is measured, and what is only emulated.

---

### 5. Sensor Fusion & EKF
A dedicated section is needed for the sensor fusion logic and EKF.
This section should explain:
- why sensor fusion is needed,
- what information is fused,
- the role of the EKF,
- state and measurement logic as appropriate,
- how visual updates interact with EKF prediction,
- gating/rejection/recovery logic if relevant,
- why this design is appropriate for the project.

All statements should be backed by sensible references where possible.
If a project-specific tuning decision is used, it must be explicitly framed as an engineering decision.

---

### 6. Visual Localization Pipeline
This section should **connect the previously explained chapters together**.
It must not read like an isolated block. Instead, it should reference earlier sections and explain how the full runtime pipeline works using those components.

This section should explain:
- how the query frame enters the system,
- how semantic information is used,
- how the search region is defined,
- how reference assets are queried,
- how candidate tiles/meta-tiles are formed,
- how geometric matching works,
- how homography-based localisation is derived,
- how the visual estimate is validated/gated,
- how the estimate is fused into the EKF,
- how file mode vs live mode differ where relevant.

This is the integration chapter and should read like the operational heart of the thesis.

---

### 7. Tools section is required
A dedicated **Tools** section/subsection is required.
This should document the main tools used in the project, such as:
- **QGIS**,
- Python,
- custom code tools/pipelines built for dataset preprocessing,
- training utilities,
- runtime scripts,
- simulation interface tooling,
- any auxiliary tools that are actually important to reproduce the workflow.

This should not become a random software list. It should describe what each tool contributed.

---

### 8. Dependency / environment requirements are required
A section/subsection is required for environment and dependency requirements.
This should include:
- Python environment / `.venv` concept,
- required libraries,
- key package groups,
- possibly a requirements file / environment file reference,
- enough reproducibility information so that the computational environment is understandable.

If the project already has `requirements.txt`, `environment.yml`, or similar, Claude should use that as the source of truth. If not, Claude should infer dependencies from imports carefully and state clearly that the list is implementation-based.

---

### 9. Mermaid diagrams are required throughout code-heavy sections
For every major section/subsection/subsubsection where code logic is explained, Claude should create **new Mermaid diagrams**.

These diagrams must:
- be based on the **current code**, not outdated diagrams,
- reflect actual input/output relationships,
- reflect functional flow,
- be technically faithful,
- be readable for humans,
- help explain code structure without replacing the text.

The old diagrams in `mermaid diagrams.zip` can be used only as a rough formatting/style reference.
They should **not** be copied blindly.

Claude should inspect the JSONL/project context to understand how the diagrams were previously intended, but then regenerate them from the current code.

### Diagram expectation
Mermaid diagrams should be used for example in:
- system overview,
- dataset generation flow,
- dataset preprocessing flow,
- training pipeline,
- simulation/live data flow,
- EKF/sensor fusion flow,
- visual localisation runtime flow,
- trace/evaluation/output flow.

Where appropriate, use input-process-output style.

---

## Evidence and citation requirement
This is a hard requirement.

### Rule
**Every technical statement should be backed by something sensible.**
That means:
- literature citation,
- official documentation,
- code-backed observation,
- experiment-backed observation,
- or clearly labelled engineering judgement.

### Acceptable phrasing examples
- “According to [reference], ...”
- “The adopted approach is consistent with ...”
- “In this work, this parameter was selected empirically based on validation behaviour ...”
- “The implementation uses X, as observed in the current codebase ...”

### Not acceptable
Claude must not write unsupported claims as if they are universal facts.
Example of bad writing:
- “This is the best method.”
- “This guarantees robustness.”
- “This approach always improves localisation.”

If a claim is based on project testing, Claude must say so explicitly.

---

## Writing style requirements
The writing must be:
- formal,
- thesis-appropriate,
- technically precise,
- well structured,
- readable,
- not fluffy,
- not repetitive.

The text should sound like a serious MSc thesis, not like software documentation pasted into LaTeX.

At the same time, code-related sections must remain concrete and traceable to the implementation.

---

## Implementation instructions for Claude Code
Claude Code should:
1. inspect the current LaTeX workspace,
2. identify where new text should be inserted,
3. preserve protected sections,
4. propose minimal structural edits where necessary,
5. generate the new LaTeX content,
6. generate/update Mermaid diagram source files if the workspace supports them,
7. integrate them into the current LaTeX structure,
8. keep references/citations coherent,
9. avoid touching the protected material unless explicitly necessary and approved.

If structural ambiguity exists, Claude should prefer:
- adding new subsections,
- adding bridge text,
- adding new figures/diagrams,
- and reordering unprotected technical sections,

rather than rewriting reviewed material.

---

## Expected output from Claude Code
Claude should produce:

### A. A plan first
Before changing files, Claude should provide a plan showing:
- what chapters/sections will be added or moved,
- which files will be edited,
- which files are protected,
- where new Mermaid diagrams will be created,
- how citations and references will be handled.

### B. Then implementation
After approval, Claude should:
- modify the LaTeX files,
- generate the required text,
- generate new Mermaid diagrams,
- keep the structure clean,
- preserve the protected sections.

### C. Final verification
After implementation, Claude should confirm:
- which files were changed,
- which protected files/sections were not touched,
- what new diagrams were created,
- whether the document compiles,
- whether references resolve,
- and whether any sections still need manual review.

---

## Final caution
Do not let the thesis become a messy dump of implementation details.
The target is a **coherent scientific/engineering document** where:
- the methodology is logically ordered,
- the code is explained clearly,
- the diagrams match the implementation,
- and the claims are properly supported.

Protected sections stay protected.
New diagrams should be newly made.
All claims need support.
