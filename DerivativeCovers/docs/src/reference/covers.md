```@meta
CurrentModule = DerivativeCovers
```

# Computing and using covers

Names that are public but not exported are part of the supported API; call
them with the module prefix, for example `DerivativeCovers.equilibrate`.

```@index
Pages = ["covers.md"]
```

## The module

```@docs
DerivativeCovers
```

## Hessian covers

```@docs
cover_hessian
cover_hybrid
forward_cover
CoverOnly
WithDiagnostics
```

## Least-squares objectives

```@docs
cover_nls
gramvariance
gramvariance!
nls_loss_tape
gn_ab
```

## Jacobian covers

```@docs
cover_jacobian
cover_jacobian_hybrid
```

## Result types

```@docs
HessianCover
HessianDiagnostics
ForwardCover
NLSCover
NLSDiagnostics
JacobianCover
JacobianDiagnostics
```

## Matrix-free products

```@docs
hvp
hvp!
jvp
jvp!
vjp
vjp!
adjoints
```

## Using covers

```@docs
scaled_spectral_radius
scaled_singular_value
equilibrate
```
