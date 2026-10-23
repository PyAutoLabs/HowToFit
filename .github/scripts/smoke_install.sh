#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

pip install ./PyAutoNerves ./PyAutoFit
# The gradient tutorials use BlackJAXNUTS. BlackJAX is a PyAutoFit optional
# dependency, so the base install above does not provide it.
pip install numba "blackjax>=1.6.2"
# tutorial_6_gradients.py also runs Nautilus, another PyAutoFit optional
# dependency; pin it as PyAutoFit's `optional` extra does.
pip install "nautilus-sampler==1.0.5"
