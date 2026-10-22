```@meta
CurrentModule = DerivativeCovers
```

# Computing and using covers

Names that are public but not exported are part of the supported API; call
them with the module prefix, for example `DerivativeCovers.cover_ratio`.

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
cover_variance
forward_cover
cover_lattice
CoverOnly
WithDiagnostics
TangentChannel
AbsoluteTangents
SignedTangents
tangent_channel
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
VarianceCover
ForwardCover
CoverLattice
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

## Measuring and using covers

```@docs
cover_ratio
bound_ratio
rank1_ratio
tightness
dependency_severity
jacobian_ratio
jacobian_representability
jacobian_minimax
scaled_spectral_radius
spectral_shrink
scaled_singular_value
jacobian_shrink
equilibrate
```
