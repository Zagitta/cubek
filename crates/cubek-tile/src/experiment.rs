//! Ablation switches for performance experiments, read on the host when a kernel expands.
//!
//! **Experiment-only: never merged.** Each switch removes real work from a kernel so a timing can
//! bound what that work costs; a kernel expanded under one computes wrong numbers. A run that sets
//! one names its own `CUBECL_ENVIRONMENT`, so no kernel compiled under a switch is reused by a run
//! without it.

/// Whether the switch `name` is set in the environment of the process expanding the kernel.
fn set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// `CUBEK_EXP_SCALES_CONSTANT`: the block-scaled leaf hands the instruction a constant scale
/// register instead of reading the operand's scales.
pub(crate) fn scales_constant() -> bool {
    set("CUBEK_EXP_SCALES_CONSTANT")
}

/// `CUBEK_EXP_SCALES_UNFILLED`: a stage's fill leaves the scales it keeps beside its values
/// unwritten.
pub(crate) fn scales_unfilled() -> bool {
    set("CUBEK_EXP_SCALES_UNFILLED")
}

/// `CUBEK_EXP_FILL_NOTHING`: a stage's fill writes nothing, values or scales; what is left is the
/// contraction and the synchronization around it.
pub(crate) fn fill_nothing() -> bool {
    set("CUBEK_EXP_FILL_NOTHING")
}

/// `CUBEK_EXP_COMPUTE_NOTHING`: the block-scaled instruction is not issued; what is left is the
/// stages' fill and the synchronization around it, the copy pipeline alone.
pub(crate) fn compute_nothing() -> bool {
    set("CUBEK_EXP_COMPUTE_NOTHING")
}
