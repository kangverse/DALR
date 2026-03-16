# Contributing to DALR

Thank you for your interest in contributing to DALR! We welcome contributions of all kinds — bug fixes, new features, documentation improvements, and more.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

1. **Fork** this repository to your GitHub account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/DALR.git
   cd DALR
   ```
3. Add the upstream remote so you can sync future changes:
   ```bash
   git remote add upstream https://github.com/kangverse/DALR.git
   ```

---

## How to Contribute

We welcome the following types of contributions:

- **Bug reports** — open a GitHub Issue using the bug report template
- **Feature requests** — open a GitHub Issue using the feature request template
- **Bug fixes** — submit a PR that references the related Issue
- **New features / experiments** — please open an Issue first to discuss the approach
- **Documentation** — fix typos, clarify confusing steps, add examples
- **Performance improvements** — profiling results and benchmarks are helpful

---

## Development Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Follow the [Getting Started](README.md#getting-started) section in the README to download the required datasets and pretrained models.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable and function names.
- Add docstrings to new public functions and classes.
- Keep functions focused — prefer small, single-purpose functions.
- Remove debugging code and unused imports before submitting.

---

## Submitting a Pull Request

1. Create a new branch from `main`:
   ```bash
   git checkout -b fix/your-descriptive-branch-name
   ```
2. Make your changes with clear, atomic commits:
   ```bash
   git commit -m "fix: correct margin variable in run_wiki_flickr.sh"
   ```
3. Keep your branch up to date with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
4. Push your branch and open a Pull Request against `main`:
   ```bash
   git push origin fix/your-descriptive-branch-name
   ```
5. Fill in the PR template — describe **what** you changed and **why**.
6. Link any related Issues in your PR description (e.g., `Closes #12`).

PRs will be reviewed as promptly as possible. Please be patient and responsive to review comments.

---

## Reporting Issues

When opening a bug report, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, PyTorch version, CUDA version)
- Relevant error messages or stack traces

---

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
