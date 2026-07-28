```@meta
CurrentModule = DerivativeCovers
```

# Internals

The sweeps, partitions, and intermediate quantities behind the entry points. These names are
documented and reachable by qualification, but carry no compatibility promise: they are the
mechanism, and a reimplementation is free to restructure them.

Also documented here are the [MatrixCovers](https://github.com/HolyLab/MatrixCovers.jl) functions
this package adds tape methods to. The functions belong to MatrixCovers; calling them unqualified
takes `using MatrixCovers` as well.

```@index
Pages = ["internals.md"]
```

```@autodocs
Modules = [DerivativeCovers]
Public = false
Private = true
```
