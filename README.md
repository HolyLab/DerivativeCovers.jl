# DerivativeCovers.jl

This repository contains two Julia packages:

- [**DerivativeCovers**](DerivativeCovers) computes certified rank-1 bounds on Hessians and
  Jacobians without forming those matrices.
- [**DerivativeCoverProblems**](DerivativeCoverProblems) provides parameterized test problems
  with reference Hessians and Jacobians for evaluating cover quality.

The test problems live in a separate package so users of DerivativeCovers do not need to install
or load them. The problems are written in generic Julia and have no dependency on the main
package.

Documentation for both packages is built from `DerivativeCovers/docs`.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/DerivativeCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/DerivativeCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/DerivativeCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/DerivativeCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
