```@meta
CurrentModule = DerivativeCovers
```

# Quantities and extension points

These names are public but not exported. Call them with the module prefix, for example
`DerivativeCovers.cover_ratio`. Like exported names, they are covered by the package's
compatibility guarantees.

This part of the API contains utilities that operate on covers from any source, along with
interfaces for extending the package.

```@index
Pages = ["reference.md"]
```

```@autodocs
Modules = [DerivativeCovers]
Public = true
Private = false
Filter = t -> !Base.isexported(DerivativeCovers, nameof(t))
```
