```@meta
CurrentModule = DerivativeCovers
```

# Internals

This page documents the sweeps, partitions, and intermediate quantities used by the public API.
They are available by qualification but are not covered by the package's compatibility
guarantees.

DerivativeCovers also adds tape methods to several
[MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl) functions. Those functions still
belong to MatrixCovers, so import that package to call them without qualification.

```@index
Pages = ["internals.md"]
```

```@autodocs
Modules = [DerivativeCovers]
Public = false
Private = true
```
