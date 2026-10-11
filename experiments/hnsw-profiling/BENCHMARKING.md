# Reproducible HNSW profiling

System controls matter for build time and search latency. They do not normally
change how counters such as hops and distance computations are interpreted, but
multithreaded construction can produce a different graph. Use the following
controls whenever results will be compared.

## Required controls

- Use the same machine, FAISS build, dataset, insertion/query order, and HNSW
  parameters for every comparison.
- Keep `--threads` fixed. Use `--threads 1` for deterministic index construction
  and single-core algorithm comparisons. Reuse that exact index for all search
  comparisons; do not rebuild it between runs.
- For scaling experiments, report every tested thread count and use the same
  multithreading policy for all candidates.
- Keep the machine idle: stop other CPU-, memory-, and I/O-intensive work, run
  on AC power, and avoid collecting results while the machine is thermally
  throttled.
- Retain the profiler's warm-up (default: 500 queries), run each timed case at
  least five times, and report the median plus a spread such as min/max or
  interquartile range. Do not compare a single run.

## This machine

The following topology was observed on this host:

- Intel Xeon Gold 5220R, one socket, 24 physical cores, SMT enabled
- CPUs `0-23` are one hardware thread from each core; CPUs `48-71` are their
  respective SMT siblings (`0/48`, `1/49`, ..., `23/71`)
- all CPUs are on NUMA node 0; node 1 contains memory but no CPUs
- about 125 GiB usable RAM, with roughly 63 GiB on each NUMA node
- `intel_pstate` frequency driver, `powersave` governor, and turbo enabled
- no isolated CPUs or `nohz_full` CPUs
- Ubuntu 22.04.5 LTS, Linux 6.8.0-60-generic

Recheck this information after hardware, BIOS, kernel, or CPU-online changes:

```bash
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE,MAXMHZ,MINMHZ,MHZ
numactl --hardware
cpupower frequency-info
```

Use `numactl` for this host instead of plain `taskset`: it pins CPUs and ensures
that memory is allocated on node 0, next to the CPUs, rather than on the
CPU-less node 1. CPU 2 is paired with SMT sibling CPU 50, so do not assign CPU
50 to another benchmark process while collecting results.

Use these commands for the standard single-thread build and search:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  numactl --physcpubind=2 --membind=0 \
  python3 experiments/hnsw-profiling/scripts/build_sift_index.py \
  --dataset sift --M 32 --ef-construction 128 --threads 1

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  numactl --physcpubind=2 --membind=0 \
  python3 experiments/hnsw-profiling/scripts/profile_sift_query.py \
  --dataset sift --M 32 --k 10 --ef-search 64 --threads 1
```

Do not add CPUs `48-71` for normal scaling tests: they are SMT siblings, not
additional physical cores. For example, use physical CPUs `2-5` for four
threads and leave their siblings `50-53` unused:

```bash
OMP_PROC_BIND=true OMP_PLACES=cores OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  numactl --physcpubind=2-5 --membind=0 \
  python3 experiments/hnsw-profiling/scripts/profile_sift_query.py \
  --dataset sift --M 32 --k 10 --ef-search 64 --threads 4
```

For other thread counts, use the same rule and set `OMP_NUM_THREADS`,
`--threads`, and the number of physical CPU IDs to the same value. Suitable
nested CPU sets are `2`, `2-3`, `2-5`, `2-9`, and `2-17` for 1, 2, 4, 8, and
16 threads. A 24-core test must use `0-23`; note that this includes CPU 0 and is
therefore not directly comparable to the smaller nested sets.

Affinity does not reserve a core: the current kernel has no isolated CPUs.
Before every timed run, confirm that the host load is low and that no other
work is using the selected cores or their SMT siblings. Dedicated core
isolation is needed if strict performance reproducibility is required.

```bash
uptime
ps -eo pid,psr,pcpu,comm --sort=-pcpu | head
```

## Frequency policy

The current `powersave` policy permits frequency changes between 1.0 and 4.0
GHz. On a dedicated host, use the performance governor to reduce ramp-up
variance, then verify it before collecting results:

```bash
sudo cpupower frequency-set --governor performance
cpupower frequency-info
```

Turbo is currently enabled. Either leave it enabled for every run or disable it
for every run; do not mix the two policies. Turbo results remain sensitive to
temperature and activity on other cores. Restore the machine's original policy
after the experiment:

```bash
sudo cpupower frequency-set --governor powersave
```

Do not change the governor on a shared host without the administrator's
approval. If it cannot be changed, retain `powersave`, record that fact, and
rely on warm-up and repeated measurements.

## Timing conditions to record

Record the CPU model, selected CPU IDs, NUMA node, thread count, SMT state,
FAISS commit/build options, compiler, dataset, HNSW parameters, and operating
system. Also record the CPU frequency policy. The performance governor can
reduce ramp-up variance when the host permits it; changing governor or turbo
settings requires administrator access and must be applied consistently to
every run.

Use the same cache policy throughout. The current profiler measures warmed
search: it performs warm-up queries and then times one batched search. If cold
cache behavior is required, define a separate procedure and label those results
as cold-cache; do not mix them with the default measurements.
