# External Academic-Skill Comparison for the Thesis Workflow

**Status:** completed comparison, 2026-08-02  
**Scope:** writing, evidence control, visual planning, and review workflow  
**Decision:** use the external repositories as methodological references; keep
the project-specific thesis workflow in a native local skill.

## 1. Sources inspected

The repositories were read from shallow snapshots rather than imported into
the project:

| Repository | Snapshot | License | Main interface |
| --- | --- | --- | --- |
| Supervisor-Skills | `aff5de9` (2026-07-16) | CC BY-NC-SA 4.0 | 11 skills and handbook |
| Academic Research Skills | `32823c3` (2026-08-02) | CC BY-NC 4.0 | Claude Code skills and agents |

Neither repository was previously present in the callable project-skill
catalog. The external installation instructions are platform-specific and do
not by themselves make a skill available to this Codex project.

## 2. What overlaps with the current catalog

| External capability | Existing local equivalent | Decision |
| --- | --- | --- |
| Evidence-bound drafting | `academic-paper` | Keep local; adopt the evidence gate |
| Logic skeleton | `academic-paper` plan | Blueprint remains the authority |
| Literature/source checks | `deep-research` + lit-review | Verify citations and bound claims |
| Figures and visual audits | `figure-designer` | Use chart and caption checks |
| Manuscript review | pre-submission + reviewer | Use at the review stage |
| Full pipeline | `academic-pipeline` + governance | Project authority wins |

Supervisor-Skills adds focused `paper-writer`, `paper-polish`,
`tech-paper-template`, and figure guidance. Academic Research Skills adds a
larger orchestrator with a Material Passport, mandatory integrity boundaries,
claim/citation audits, and process records. These are valuable patterns, but
they substantially overlap with the installed academic skills and are written
for different runtimes.

## 3. What is genuinely missing for this project

The generic skills do not encode the project's complete writing contract in one
callable procedure. In particular, they do not jointly enforce:

1. Vietnamese meaning drafting followed by user confirmation and original
   English rewriting;
2. the thesis Evidence Map and visual inventory as required inputs;
3. the 6-fps source-time versus 30-fps playback distinction;
4. the boundary between behavioral-deviation screening and diagnosis;
5. the provisional status of current review artifacts and final metrics;
6. the main posture experiment and its separate authority requirement;
7. the inherited-data versus personal-contribution distinction.

This gap is now covered by the project-local skill:

`.agents/skills/thesis-evidence-writing/SKILL.md`

It is registered as an explicit (`implicit: false`) project skill so it can be
called intentionally for thesis work without hijacking unrelated code or data
tasks. Its validator reports `Skill is valid!`.

## 4. Adopted working principles

The thesis workflow will use the following principles, expressed in original
project-specific instructions rather than copied repository text:

- establish an evidence ledger before drafting;
- cap wording strength at the evidence strength;
- distinguish confirmed results, protocol, and planned work;
- rewrite from meaning rather than translate sentence by sentence;
- preserve the user's scientific meaning during language editing;
- treat figures, tables, real frames, and reproducible plots as evidence;
- require a human meaning checkpoint before English conversion;
- stop on authority conflicts, unreadable evidence, or a missing visual anchor.

## 5. Principles not imported

The project does not copy external `SKILL.md` files, agents, templates, images,
or scripts. It also does not import Claude-specific symlink layouts,
autonomous experiment execution, generic top-venue claims, or external
Material Passport state. Any future direct adaptation would require a separate
license and attribution review; the current local skill is an original,
project-scoped workflow.

## 6. Operating sequence for the thesis

1. Use `thesis-evidence-writing` to define the subsection contract, evidence
   ledger, Vietnamese draft, and visual anchor.
2. Let the user confirm the technical meaning and unresolved terminology.
3. Use the same skill to write original English prose and the caption/table
   text, with pending items explicitly marked.
4. Use `academic-paper` for chapter architecture, citations, and formatting.
5. Use `figure-designer` for the solution overview, real-data figures, and
   result plots.
6. Use `pre-submission-reviewer` and `academic-paper-reviewer` after the
   relevant evidence and manuscript sections are complete.

This sequence does not authorize data relabeling, model execution, source
correction, or promotion of an unbound metric into a thesis result.
