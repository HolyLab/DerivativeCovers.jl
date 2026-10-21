```@meta
CurrentModule = DerivativeCovers
```

# Internals

This page documents the sweeps, partitions, and intermediate quantities used to
implement the public API. They can be called with the `DerivativeCovers.`
prefix, but they are not part of the supported API and may change without
notice.

DerivativeCovers also adds methods for recorded tapes to several functions of
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl), such as
`gramcover`. Those functions belong to MatrixCovers, so load that package to
call them without qualification.

```@index
Pages = ["internals.md"]
```

```@autodocs
Modules = [DerivativeCovers]
Public = false
Private = true
```
