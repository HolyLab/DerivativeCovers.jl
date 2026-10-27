```@meta
CurrentModule = DerivativeCovers
```

# Extension interface

These names define how programs are recorded, how local covers are chosen, and how new
operations and number types are supported. Apart from `Tape`, `TVar`, `tape_function`, and
`FVar`, they are public but not exported; call them with the module prefix, for example
`DerivativeCovers.unarynode`.

```@index
Pages = ["extending.md"]
```

## Choosing among local covers

```@docs
split_block
split_ratio
guards_any_block
sensitivity_scales
split_scales
SplitScale
dead_slot
forward_scale
```

## Recording programs

```@docs
Tape
TVar
tape_function
FVar
CoverVar
value
isconstant
constantvar
nnodes
natoms
noderef
```

## Adding operations

```@docs
pushnode!
unarynode
binarynode
dotnode
linearnode
constantnode
atom_slots
check_finite_partials
check_values
```

## Second-derivative forms

```@docs
Curvature
NoCurvature
UnaryCurvature
BinaryCurvature
DenseCurvature
PairedCurvature
DeclaredCurvature
cover_slots
cover_slots!
bound_contribution!
curvature_apply!
has_exact_partials
CoverSite
TapeSite
site_label
site_values
```

## Sub-maps and composition

```@docs
submap
declare_submap
submapnode
compose_jacobian_cover
compose_hessian_cover
```

## Set-valued numbers

```@docs
magnitude
magtype
boundtype
Upward
decided_sign
decided_one
separation
```
