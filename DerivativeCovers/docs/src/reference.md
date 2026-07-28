```@meta
CurrentModule = DerivativeCovers
```

# Quantities and extension points

Names that are `public` but not exported: reached by qualification
(`DerivativeCovers.cover_ratio`), and covered by the same compatibility promise as the exports.
Two kinds of name qualify — the quantities defined on a cover from any source, and the contracts
that outside code implements — and neither can be disturbed by a reimplementation of the sweeps.

```@index
Pages = ["reference.md"]
```

```@autodocs
Modules = [DerivativeCovers]
Public = true
Private = false
Filter = t -> !Base.isexported(DerivativeCovers, nameof(t))
```
