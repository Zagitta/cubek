use cubecl::{
    prelude::barrier::{Barrier, copy_async_commit},
    prelude::*,
};

use crate::multi_level::{
    components::global::{SharedGlobalMatmulConfig, read::SyncStrategy},
    definition::MatmulTypes,
};

/// Asynchronous barrier for `async_memcpy`
pub struct AsyncBarrier {}

#[cube]
impl SyncStrategy for AsyncBarrier {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulTypes>(
        barrier: &Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig,
    ) {
        barrier.arrive_and_wait();
    }
}

/// Completes `copy_async` loads through a commit group: each unit waits for its own copies,
/// then the cube synchronizes so every unit sees the whole stage, as sm_80 GEMMs do
/// (`cp.async.wait_group` + `bar.sync`). Waiting on the barrier instead is a polling loop over
/// shared memory. The barrier stays the type of the specialized loaders, whose producer planes
/// complete their copies on it.
pub struct AsyncCopy {}

#[cube]
impl SyncStrategy for AsyncCopy {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(CUBE_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulTypes>(
        _barrier: &Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig,
    ) {
        copy_async_commit().wait();
        sync_cube();
    }
}
