//! What reading through a page table costs a decode step: the walk of an attention decode, read
//! dense and through a table ([`TileArg::tile_paged`]) by the same kernel, so any gap is the
//! paged read.
//!
//! Every cube is one plane walking its run of one head's positions four keys a step, as a shuffle
//! plane does: sixteen lanes a key, two keys in flight, each lane reading one line of each. One
//! body sums the keys' and values' lines; the other scores each attended key, its dot with a query
//! of ones summed over its sixteen lanes, and weights its value by it.
//!
//! A measurement, ignored by default:
//!
//! ```sh
//! cargo test -p cubek-tile --release --no-default-features --features std,cubecl/cuda --test lib \
//!     -- paged_probe --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use cubecl::{client::Client, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_test_utils::TestInput;
use cubek_tile::launch::Grid;
use cubek_tile::*;

const B: Axis = Axis(0);
const H: Axis = Axis(1);
const S: Axis = Axis(2);
const D: Axis = Axis(3);

/// Qwen3-8B's key heads and head dim.
const HEADS: usize = 8;
const DIM: usize = 128;
const WIDTH: usize = 8;
/// Cubes sharing a head's positions, each a run of its steps.
const CUBES: usize = 32;
/// A shuffle plane's walk: sixteen lanes a key, two teams of them, two keys each in flight.
const LANES_PER_KEY: usize = DIM / WIDTH;
const TEAMS: usize = 2;
const IN_FLIGHT: usize = 2;
const STEP: usize = TEAMS * IN_FLIGHT;
const LANES: u32 = (TEAMS * LANES_PER_KEY) as u32;
/// `f32`s of shared memory a cube reserves only to hold an SM to about eight cubes, as a
/// shuffle routine's registers do.
const RESERVED: usize = 2048;

/// What a step does with the lines it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Body {
    /// Sums the keys' and the values' lines.
    KeysValues,
    /// Each attended key's dot with a query of ones, summed over its lanes, weighting its value.
    Scores,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn decode_probe_kernel<E: Float, N: Size>(
    key: &TileArg<'_, E>,
    value: &TileArg<'_, E>,
    pages: ComptimeOption<&Tensor<u32>>,
    out: &mut Tensor<Vector<f32, N>>,
    attended: u32,
    space: Partitioning,
    #[comptime] body: Body,
    #[comptime] ahead: bool,
    #[define(E)] _elem: ElemType,
) {
    let mut held = Shared::<[f32]>::new_slice(RESERVED);
    held[UNIT_POS as usize] = 0.0;
    let k = key.tile_paged(&space, pages);
    let v = value.tile_paged(&space, pages);
    let lane = UNIT_POS_PLANE as usize;
    let (team, part) = (lane / LANES_PER_KEY, lane % LANES_PER_KEY);
    let mut sum = Vector::<f32, N>::cast_from(0u32);
    let run = space.walk();
    // Under `ahead`, two steps' windows are cut before either is read, so the second step's page
    // table load overlaps the first step's reads.
    let pairs = run.total() / 2;
    for i in 0..pairs {
        if comptime!(ahead) {
            let (a, b) = (run.region(2 * i), run.region(2 * i + 1));
            let (ka, va, kb, vb) = (k.at(&a), v.at(&a), k.at(&b), v.at(&b));
            sum += step_sum::<E, N>(&ka, &va, a.origin(S), team, part, attended, body);
            sum += step_sum::<E, N>(&kb, &vb, b.origin(S), team, part, attended, body);
        } else {
            let a = run.region(2 * i);
            let (ka, va) = (k.at(&a), v.at(&a));
            sum += step_sum::<E, N>(&ka, &va, a.origin(S), team, part, attended, body);
            let b = run.region(2 * i + 1);
            let (kb, vb) = (k.at(&b), v.at(&b));
            sum += step_sum::<E, N>(&kb, &vb, b.origin(S), team, part, attended, body);
        }
    }
    out[ABSOLUTE_POS] = sum + Vector::cast_from(held[UNIT_POS as usize]);
}

/// One step's lines as `body` takes them, the step's keys starting at `first`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn step_sum<E: Float, N: Size>(
    key_window: &Tile,
    value_window: &Tile,
    first: usize,
    team: usize,
    part: usize,
    attended: u32,
    #[comptime] body: Body,
) -> Vector<f32, N> {
    let keys = key_window.flat::<E, N>();
    let values = value_window.flat::<E, N>();
    let mut sum = Vector::<f32, N>::cast_from(0u32);
    #[unroll]
    for c in 0..IN_FLIGHT {
        let at = c * TEAMS + team;
        let line = at * LANES_PER_KEY + part;
        let seen = ((first + at) as u32) < attended;
        match comptime!(body) {
            Body::KeysValues => {
                sum += Vector::cast_from(keys.read(line));
                sum += Vector::cast_from(values.read(line));
            }
            Body::Scores => {
                let mut k_line = Vector::<f32, N>::cast_from(0u32);
                let mut v_line = Vector::<f32, N>::cast_from(0u32);
                if seen {
                    k_line = Vector::cast_from(keys.read(line));
                    v_line = Vector::cast_from(values.read(line));
                }
                let mut score = Monoid::reduce::<f32, N>(k_line, WIDTH, Monoid::Sum);
                score += plane_shuffle_xor(score, 1);
                score += plane_shuffle_xor(score, 2);
                score += plane_shuffle_xor(score, 4);
                score += plane_shuffle_xor(score, 8);
                sum += v_line * Vector::cast_from(score);
            }
        }
    }
    sum
}

/// How the probe reads the keys and values.
#[derive(Clone, Copy, Debug)]
enum Read {
    Dense,
    /// Through a table of pages this many positions long, a sequence's pages in reverse order.
    Paged(usize),
    /// [`Paged`](Self::Paged), the pages in order: the dense layout cut in pages.
    InOrder(usize),
}

/// What position `s` holds: a value that changes every few keys, so a window read twice in place
/// of another changes the sum.
fn value(s: usize) -> f64 {
    (s / 4 % 5 + 1) as f64
}

/// What every cube's sums add up to over `keys` positions, `attended` of them attended.
fn expected(body: Body, keys: usize, attended: usize) -> f64 {
    let lines = (HEADS * DIM) as f64;
    let over = |n: usize, f: &dyn Fn(f64) -> f64| (0..n).map(|s| f(value(s))).sum::<f64>();
    match body {
        Body::KeysValues => 2.0 * lines * over(keys, &|v| v),
        Body::Scores => lines * DIM as f64 * over(attended, &|v| v * v),
    }
}

struct Probe {
    launcher: Launcher,
    key: TensorHandle,
    value: TensorHandle,
    read: Read,
    pages: Option<TensorHandle>,
    out: TensorHandle,
}

fn probe(client: &Client, keys: usize, read: Read) -> Probe {
    let space = Space::new(&[(B, 1), (H, HEADS), (S, keys), (D, DIM)]);
    let levels = Levels::leaf(&[(B, 1), (H, 1), (S, STEP)])
        .walk_every(&[S])
        .cubes(&[S])
        .across(S, CUBES)
        .interleaved(S)
        .batches(&[B, H])
        .build();
    let partitioning = Partitioning::new(space.clone(), levels);
    let from_levels =
        Launcher::new(client, partitioning.clone(), &space, Grid::FromLevels).unwrap();
    let launcher = Launcher::new(
        client,
        partitioning,
        &space,
        Grid::Stated {
            cube_count: from_levels.cube_count(),
            cube_dim: CubeDim::new_1d(LANES),
        },
    )
    .unwrap();
    // Position `s` holds `value(s)` in every head and lane, wherever its page lies.
    let (buffer, values, table) = match read {
        Read::Dense => (
            vec![1, HEADS, keys, DIM],
            (0..HEADS * keys * DIM)
                .map(|i| value(i / DIM % keys) as f32)
                .collect::<Vec<_>>(),
            None,
        ),
        Read::Paged(page) | Read::InOrder(page) => {
            let each = keys.div_ceil(page);
            let reversed = matches!(read, Read::Paged(_));
            let column = |slot: usize| if reversed { each - 1 - slot } else { slot };
            let values = (0..each * HEADS * page * DIM)
                .map(|i| {
                    let (slot, row) = (i / (HEADS * page * DIM), i / DIM % page);
                    value(column(slot) * page + row) as f32
                })
                .collect();
            let table: Vec<f32> = (0..each).map(|j| column(j) as f32).collect();
            (vec![each, HEADS, page, DIM], values, Some((each, table)))
        }
    };
    let side = || {
        TestInput::builder(client.clone(), Shape::from(buffer.clone()))
            .dtype(half::f16::elem_type_native())
            .custom(values.clone())
            .generate_without_host_data()
    };
    let pages = table.map(|(each, table)| {
        TestInput::builder(client.clone(), Shape::new([1, each]))
            .dtype(u32::elem_type_native())
            .custom(table)
            .generate_without_host_data()
    });
    let units = match from_levels.cube_count() {
        CubeCount::Static(x, y, z) => (x * y * z * LANES) as usize,
        _ => unreachable!("the probe's grid is static"),
    };
    let out = TestInput::builder(client.clone(), Shape::new([units * WIDTH]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();
    Probe {
        launcher,
        key: side(),
        value: side(),
        read,
        pages,
        out,
    }
}

fn launch(client: &Client, p: &Probe, body: Body, ahead: bool, attended: usize) {
    let bound = |side: &TensorHandle| {
        let arg = p
            .launcher
            .arg(side.clone().binding())
            .axes(&[B, H, S, D])
            .vectorize(WIDTH);
        match p.read {
            Read::Dense => arg.build(),
            Read::Paged(_) | Read::InOrder(_) => arg.paged_covering(B, S).build(),
        }
        .unwrap()
    };
    let pages = match &p.pages {
        Some(pages) => ComptimeOptionArgs::Some(pages.clone().binding().into_tensor_arg()),
        None => ComptimeOptionArgs::None,
    };
    unsafe {
        decode_probe_kernel::launch_unchecked(
            client,
            p.launcher.cube_count(),
            p.launcher.cube_dim(),
            WIDTH,
            bound(&p.key).arg(),
            bound(&p.value).arg(),
            pages,
            p.out.clone().binding().into_tensor_arg(),
            attended as u32,
            p.launcher.partitioning_arg(),
            body,
            ahead,
            half::f16::elem_type_native(),
        );
    }
}

/// Microseconds a launch, the best of several rounds of back-to-back launches.
fn time(client: &Client, p: &Probe, body: Body, ahead: bool, attended: usize) -> f64 {
    const LAUNCHES: usize = 20;
    for _ in 0..3 {
        launch(client, p, body, ahead, attended);
    }
    cubecl::future::block_on(client.sync()).unwrap();
    (0..7)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..LAUNCHES {
                launch(client, p, body, ahead, attended);
            }
            cubecl::future::block_on(client.sync()).unwrap();
            start.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64
        })
        .fold(f64::MAX, f64::min)
}

/// One launch of keys and values at 4096 keys, dense or through one page, for a PTX dump:
/// `PAGED_PROBE_DUMP=dense` or `=paged`.
#[test]
#[ignore = "a dump"]
fn paged_probe_dump() {
    let client = cubecl::test_device().client();
    let read = match std::env::var("PAGED_PROBE_DUMP").as_deref() {
        Ok("paged") => Read::Paged(4096),
        _ => Read::Dense,
    };
    let p = probe(&client, 4096, read);
    launch(&client, &p, Body::KeysValues, false, 4091);
    cubecl::future::block_on(client.sync()).unwrap();
}

#[test]
#[ignore = "a measurement"]
fn paged_probe() {
    let client = cubecl::test_device().client();
    for keys in [4096, 32768] {
        // Three quarters attended, so a scored read through the wrong pages masks other positions
        // and changes its sum.
        let attended = keys * 3 / 4 + 3;
        let reads = [
            Read::Dense,
            Read::Paged(keys),
            Read::Paged(64),
            Read::InOrder(64),
            Read::Paged(16),
            Read::InOrder(16),
        ];
        let probes: Vec<_> = reads
            .iter()
            .map(|&r| (r, probe(&client, keys, r)))
            .collect();
        for (body, ahead) in [
            (Body::KeysValues, false),
            (Body::KeysValues, true),
            (Body::Scores, false),
            (Body::Scores, true),
        ] {
            // Every read of every position once: a wrong page or a skipped step shows here.
            for (read, p) in &probes {
                launch(&client, p, body, ahead, attended);
                let bytes = client.read_one_unchecked(p.out.handle.clone());
                let total: f64 = f32::from_bytes(&bytes).iter().map(|&x| x as f64).sum();
                assert_eq!(
                    total,
                    expected(body, keys, attended),
                    "{body:?} {read:?} at {keys} keys"
                );
            }
            let dense = time(&client, &probes[0].1, body, ahead, attended);
            for (read, p) in &probes {
                let us = time(&client, p, body, ahead, attended);
                eprintln!(
                    "{:<10} {:<6} keys {keys:>6}  {:<14} {us:>8.1} us  {:>+5.0}%",
                    format!("{body:?}"),
                    if ahead { "ahead" } else { "inline" },
                    format!("{read:?}"),
                    (us / dense - 1.0) * 100.0
                );
            }
        }
    }
}
