# MR-IQA-2 Paper

This directory contains the arXiv manuscript and standalone supplementary
materials for **MR-IQA-2: Faithful Image Quality Reflection via Fine-Grained
Credit Assignment**.

## Build

Compile the merged arXiv manuscript with:

```bash
latexmk -pdf AnonymousSubmission2027_ExperimentAudit.tex
```

Compile the standalone supplementary materials with:

```bash
latexmk -pdf SupplementaryMaterials_ExperimentAudit.tex
```

Both commands should be run from this directory. The figures used by the
manuscripts are stored in `Figures/`.
