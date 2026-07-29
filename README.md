# DerivativeCovers.jl

This repository holds two packages, each in its own subdirectory:

- [**DerivativeCovers**](DerivativeCovers) — rank-1 covers of a program's Hessian at about twice
  the cost of a gradient.
- [**DerivativeCoverProblems**](DerivativeCoverProblems) — parameterized families of programs,
  with reference Hessians and Jacobians, for measuring how loose a cover is.

The corpus is a separate package because it depends on nothing in the library: its programs are
written in generic Julia and are taped by virtue of that. Keeping them apart means a user of the
covers never loads the fixtures, and the corpus can be versioned and cited on its own.

Documentation for both is built from `DerivativeCovers/docs`.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
