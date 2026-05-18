# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CODE_OF_CONDUCT.md` adopting the Contributor Covenant v2.1
- `CITATION.cff` for GitHub's "Cite this repository" feature
- This changelog

### Changed
- Consolidated the two model files: the legacy `src/model_dalr.py` was
  removed and `src/model_modified.py` (the version used by the training
  pipeline) was renamed to `src/model_dalr.py`.

## [0.1.0] - 2025-03-20

Initial public release accompanying the ACL 2025 Findings paper
*"DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning."*

### Added
- DALR model implementation (`src/model_dalr.py`)
- Training pipeline for Wiki + Flickr30k and Wiki + MS-COCO (`src/train_mix.py`, `scripts/`)
- SentEval-based evaluation (`src/evaluation.py`, `src/utils.py`)
- Cross-modal alignment loss and ranking-based intra-modal distillation
- Align / uniform loss metrics
- English and Chinese READMEs
- Contribution guide, bug report and feature request issue templates, PR template

[Unreleased]: https://github.com/kangverse/DALR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kangverse/DALR/releases/tag/v0.1.0
