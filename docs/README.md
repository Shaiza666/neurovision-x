# Documentation map

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

**Start here:** repository [README](../README.md) (status, install, dashboards) → [dataset.md](dataset.md) (Synapse; no auto-download) → [training.md](training.md) / [evaluation.md](evaluation.md) → [research_report.md](research_report.md).

Phase 8 portfolio pack (numbers match the documentation checklist):

1. GitHub README — repository root `README.md`
2. Abstract — [abstract.md](abstract.md)
3. Introduction — [research_report.md](research_report.md) §1
4. Related work — [literature_review.md](literature_review.md), [research_report.md](research_report.md) §2
5. Research gap and hypotheses — [research_gap.md](research_gap.md), [research_hypotheses.md](research_hypotheses.md)
6. Methodology — [research_report.md](research_report.md) §4
7. Mathematical formulation — [proposed_method.md](proposed_method.md), [metrics.md](metrics.md)
8. Dataset description — [research_report.md](research_report.md) §6, `data/raw/README.md`
9. Experimental protocol — [experimental_protocol.md](experimental_protocol.md)
10. Results and discussion — [experimental_results.md](experimental_results.md) (on-disk JSON only)
11. Ablation study — [research_report.md](research_report.md) §9 (**not measured** on BraTS)
12. Limitations — [research_report.md](research_report.md) §10
13. Conclusion and future work — [research_report.md](research_report.md) §11
14. References — [references.md](references.md) (IEEE), [references.bib](references.bib)
15. System architecture — [architecture.md](architecture.md)
16. Reproducibility — [reproducibility.md](reproducibility.md)
17. Installation and usage — [installation.md](installation.md)
18. Presentation slides — [presentation.md](presentation.md) (Marp markdown)
19. Demonstration script — [demo_script.md](demo_script.md)

Portfolio summary (contribution, implementation, CV relevance): [portfolio_summary.md](portfolio_summary.md).

| Document | Contents |
| --- | --- |
| [research_report.md](research_report.md) | Abstract through conclusion: the portfolio write-up |
| [architecture.md](architecture.md) | System diagram and module map |
| [installation.md](installation.md) | Environment, install, and command index |
| [dataset.md](dataset.md) | Official BraTS acquisition (Synapse; no auto-download) |
| [training.md](training.md) | Smoke vs official A0–A5; hardware gate |
| [evaluation.md](evaluation.md) | Dice/HD95/ECE JSON+CSV; H1 compare_runs |
| [reproducibility.md](reproducibility.md) | Seeds, splits, run folders, honesty rules |
| [literature_review.md](literature_review.md) | Phase 1 verified related work |
| [research_gap.md](research_gap.md) | Open problems (not claimed solutions) |
| [research_hypotheses.md](research_hypotheses.md) | H1–H3 (untested on BraTS) |
| [proposed_method.md](proposed_method.md) | Mathematics of established components |
| [experimental_protocol.md](experimental_protocol.md) | Ablation arms A0–A5 (**awaiting approval**) |
| [evaluation_protocol.md](evaluation_protocol.md) | A/B/C comparison plan |
| [experimental_results.md](experimental_results.md) | On-disk measurements only |
| [experiment_summary.md](experiment_summary.md) | Short status of what was actually measured |
| [metrics.md](metrics.md) | Dice, HD95, ECE, NaN policy |
| [explainability.md](explainability.md) | Grad-CAM and overlay limits |
| [demo.md](demo.md) | Research Streamlit dashboard |
| [demo_script.md](demo_script.md) | Live demonstration walkthrough |
| [presentation.md](presentation.md) | Research slides (Marp markdown) |
| [portfolio_summary.md](portfolio_summary.md) | Contribution, implementation, CV relevance |
| [references.md](references.md) | IEEE-style bibliography |
| [references.bib](references.bib) | BibTeX |

BraTS A/B/C Dice, HD95, and ECE are **not measured**. Do not treat smoke-run numbers as challenge results.
