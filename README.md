# `latticesort`

`latticesort` is a zero-dependency, 100% safe Rust sorting library for primitive integers (`i32`, `u32`, `i64`, `u64`) and floating-point slices (`f32`, `f64`). Developed through an autonomous experimental research campaign (`autoresearch`) and hardened against an adversarial red-team suite, `latticesort` achieves a **4.79x to 5.68x geometric mean speedup** at $N = 1{,}000{,}000$, a **4.02x to 4.08x geometric mean speedup** at $N = 10{,}000{,}000$, and up to **7.68x speedup** at $N = 1{,}000{,}000{,}000$ ($4.00\text{ GB}$) over Rust's standard library `std::slice::sort_unstable` (`ipnsort` / pattern-defeating quicksort).

---

## Why the Name `LatticeSort`?

1. **Order Theory and Abstract Algebra**: In discrete mathematics and Computer Science, a **lattice** is a partially ordered set $(L, \le)$ in which every pair of elements $(a, b)$ has a unique supremum (join, $a \vee b = \max(a, b)$) and a unique infimum (meet, $a \wedge b = \min(a, b)$). Every totally ordered numeric domain sorted by this library (`i32`, `u32`, `i64`, `u64`, and total-order `f32`/`f64`) forms a distributive lattice.
2. **Solid-State Physics and Hardware Cache Geometry**: In physics, a **crystal lattice** is a periodic array of fixed spatial sites where particles settle into minimum-energy equilibrium positions. Instead of sorting via millions of pairwise element-to-element duels, `latticesort` projects keys directly into fixed-address coordinate slots inside L1 (Level 1 Cache, the fastest cache memory core-local to the processor) and L2 (Level 2 Cache, the intermediate cache memory between L1 and shared cache).
3. **In Plain English**: Traditional sorting works like a line of blindfolded people trying to line up by height by comparing themselves two at a time. `LatticeSort` works like a hotel mailroom wall of numbered pigeonholes (a **lattice**): each number's own mathematical value tells it the exact pigeonhole where it belongs, skipping pairwise duels altogether.

---

## Part 1: How `LatticeSort` Works in Plain English (Layman's Guide)

### 1. The "Quick Glance" and the Zipper Merge (*Bitonic Run Merger*)
Real-world lists often arrive with an existing shape: stock prices that rose until midday and then fell (a wave), reversed logs, or database tables that are already 90% sorted with a few new rows appended at the end.
* **What standard algorithms do**: Chop the entire list into pieces and start comparing items from scratch.
* **What `LatticeSort` does**: It takes a quick glance at just **8 evenly spaced spots** in the list. When it sees a rising-then-falling wave (or a falling-then-rising valley), it grabs both ends of the wave and pulls them together like a **zipper**, merging the two halves in a single pass.

### 2. The 1024-Slot Egg Carton and the "Two-Coin Trick" (*Dual-Sentinel L1 Hash Aggregation*)
Huge real-world datasets frequently repeat a small set of distinct values (customer ages, ratings from 1 to 100, status codes, or sensor tiers).
* **What standard algorithms do**: Shuffle the same duplicate numbers back and forth millions of times.
* **What `LatticeSort` does**: It places a tiny **1024-slot counting tray** inside the processor's fastest memory (L1) and simply tallies how many times each value appears: *"the number 7 appeared 80,000 times; the number 12 appeared 90,000 times"*. At the end, it writes out eighty thousand 7s followed by ninety thousand 12s in one contiguous sweep.
* **The Two-Coin Trick (Dual-Sentinel Invariant)**: Normally, to know whether a slot in a counting tray is empty or already holding a number, the computer must check a separate "occupied" checklist on every single item. `LatticeSort` eliminates that checklist check: it pre-fills every slot in the tray with the number `0`, except for the single slot where `0` itself belongs, where it places the number `1`. Because no slot starts out holding its own true owner, the processor immediately knows a slot is valid whenever the number inside matches the number being read.

### 3. The Numbered Ruler for Clustered Numbers (*Cache-Bounded Dense Range Counting*)
Sometimes a list has thousands of different numbers, but they all sit within a compact range (such as salaries, temperatures, or product prices between \$10 and \$5,000).
* **What `LatticeSort` does**: It measures the distance between the smallest and largest number in the list. When that distance fits inside fast processor memory, it lays down a numbered ruler from the minimum value to the maximum value, tallies how many items land on each millimeter mark, and reads the ruler from left to right. On financial and Pareto distributions, this is **11x faster** than Rust's standard sort.

### 4. The Single-Pass Post Office Sorter that Skips Identical Zip Codes (*Fused LSD Radix Sort with Byte Skip*)
When numbers are completely random and span billions of values, `LatticeSort` acts like an automated postal sorting machine that sorts letters digit by digit (first by street, then city, then state, then country), a technique known as LSD (Least Significant Digit, a radix sorting order that processes bytes from lowest to highest significance) radix sort.
* **What classic Radix Sort does**: Reads all 1 million letters to count the streets, moves all 1 million letters; reads all 1 million letters again to count the cities, moves all 1 million letters; repeating 8 full trips across memory.
* **What `LatticeSort` does**:
  1. **One-Trip Quadruple Tally**: Reads the list **once** at the start and tallies all 4 byte positions (street, city, state, country) at the same time.
  2. **Skipping Identical Digits**: If the tally shows that all 1 million letters belong to the same country and state (which happens constantly with timestamps from the same year or positive integers in a bounded range), `LatticeSort` **skips those sorting passes in zero seconds**, cutting memory traffic in half.

### 5. Dividing 10 Million Items into 256 "Neighborhoods" (*MSB + L2 Hybrid Partitioning*)
Sorting 10 million items scattered across a giant warehouse takes a long time because workers spend all their time walking back and forth across the warehouse floor. In a computer, that distant warehouse is RAM (Random-Access Memory, the main system memory outside the processor), while the workbench right in front of the worker is L2 cache.
* **What `LatticeSort` does**: For lists of 1.5 million items or more, it first groups the items by their highest byte, called the MSB (Most Significant Byte, the highest-order byte of a multi-byte word that partitions keys into coarse buckets), into **256 smaller neighborhood boxes**. Each neighborhood box is small enough (~156 KB) to fit completely on the processor's L2 workbench, so the processor finishes each box at top speed without walking back to RAM, or hands different boxes to multiple processor cores simultaneously (**5.05x faster** than standard Rust).

---

## Part 2: Mathematical Foundations, Formal Invariants, and Technical Architecture

### Stage 1: Bitonic Envelope & Structural Run Merger (`try_bitonic_run_merge`)

Given an input slice $A[0 \dots n-1]$, an 8-point constant-time stride probe evaluates the discrete derivative signs at indices $p_k = k \cdot \lfloor n / 8 \rfloor$ for $k \in \{1, \dots, 7\}$:

$$s_k = \operatorname{sgn}(A[p_k] - A[p_{k-1}]) \in \{-1, 0, +1\}$$

If the number of sign inversions $\sum_{k=2}^{7} \mathbb{I}(s_k \neq 0 \land s_k \neq s_{k-1}) > 1$, the stage aborts in $O(1)$ time (7 comparisons). Otherwise, it scans for the monotone inflection index $m$:
* **Strictly Ascending ($m = n - 1$)**: Returns immediately in $O(n)$ with 0 writes.
* **Strictly Descending ($m = 0$, valley at $n - 1$)**: Reverses $A$ in-place via $\lfloor n/2 \rfloor$ swaps.
* **Crest-to-Trough Bitonic Envelope ($\cap$, $A[0] \le \dots \le A[m]$ and $A[m+1] \ge \dots \ge A[n-1]$)**: Merges the ascending left slice $A[0 \dots m]$ (indexed forward by $i = 0 \to m$) and the descending right slice $A[m+1 \dots n-1]$ (indexed backward by $j = n-1 \to m+1$) in a single linear pass:

$$T_{\text{bitonic}}(n) = O(n) \text{ comparisons and } 1 \text{ sequential write pass}$$

* **Plateau Snap-Back Invariant**: When $m \ge \frac{3}{4}n$ (long sorted prefix followed by an unsorted tail) and $A[0] = A[\lfloor m/2 \rfloor] = c$, the prefix contains a constant plateau. If $A[m] > c$, the split index snaps back to $m' = \max \{ idx \le m \mid A[idx] = c \}$, sorts the tail $A[m'+1 \dots n-1]$, and splices the entire constant block $c$ into position in a single `copy_within` + `fill` operation without allocating a merge buffer.

---

### Stage 2: Dual-Sentinel Direct-Mapped Hash Aggregation (`try_sentinel_hash_aggregation`)

To detect low-cardinality inputs ($K \le 384$ unique keys) while defeating localized entropy traps, `latticesort` samples 128 elements using 5 contiguous 16-element quartile bursts at anchors $q \in \{0.1n, 0.3n, 0.5n, 0.7n, n-16\}$ combined with 48 golden-ratio stride samples:

$$\text{stride} = \left\lfloor \frac{n \cdot 2654435761}{2^{32}} \right\rfloor \lor 1, \quad \varphi^{-1} \approx \frac{2654435761}{2^{32}}$$

#### Multiplicative Hash Family
For a 32-bit key $x \in \mathbb{Z}_{2^{32}}$ and a candidate odd multiplier $\mu \in \{ \text{0x9e3779b9}, \text{0x85ebca6b}, \text{0xc2b2ae35}, \text{0x27d4eb2f} \}$, the 10-bit slot index in the 1024-entry L1 table is:

$$h_{\mu}(x) = \left( ((x \oplus (x \gg 15)) \cdot \mu) \bmod 2^{32} \right) \gg 22 \in \{0, \dots, 1023\}$$

The algorithm selects the first multiplier $\mu^*$ that produces zero collisions across the unique keys of the 128-element probe.

#### Formal Proof of the Dual-Sentinel Invariant
Let $s_0 = h_{\mu^*}(0)$ be the home slot of key $d_0 = 0$, and choose $d_1 = \min \{ v \in \mathbb{Z}^+ \mid h_{\mu^*}(v) \neq s_0 \}$. Initialize the 1024-slot key array $K[0 \dots 1023]$ as:

$$K[s] = \begin{cases} d_1 & \text{if } s = s_0 \\ d_0 & \text{if } s \neq s_0 \end{cases}$$

**Theorem**: Prior to any insertion, $\forall x \in \mathbb{Z}_{2^{32}}$, let $s = h_{\mu^*}(x)$; then $K[s] \neq x$.
* *Case 1 ($s = s_0$)*: By definition of $s = h_{\mu^*}(x) = s_0$ and $h_{\mu^*}(d_1) \neq s_0$, we have $x \neq d_1 = K[s_0]$.
* *Case 2 ($s \neq s_0$)*: Since $h_{\mu^*}(d_0) = s_0 \neq s = h_{\mu^*}(x)$, we have $x \neq d_0 = K[s]$.

Consequently, a direct hit $K[h_{\mu^*}(x)] == x$ on the home slot is a **necessary and sufficient condition** that $x$ has already been registered in its home slot, eliminating the occupancy bitmap check `used[s]` from the fast path of the inner loop:

$$T_{\text{hash}}(n, K) = O(n + K \log K), \quad \text{Space} = 8\text{ KB on stack (L1)}$$

---

### Stage 3: Cache-Bounded Dense Range Counting (`try_dense_range_count`)

The 128-element probe from Stage 2 yields a lower bound on the global span:

$$\Delta_{\text{probe}} = \max(P) - \min(P) \le \Delta_{\text{global}} = \max(A) - \min(A)$$

If $\Delta_{\text{probe}} > \min(n, 1{,}048{,}576)$, Stage 3 is bypassed in $O(1)$ time with zero memory reads. Otherwise, if $\Delta_{\text{global}} \le \min(n, 1{,}048{,}576)$, an offset frequency array $D[0 \dots \Delta_{\text{global}}]$ is populated inside L2/L3 (Level 3 Cache, the shared last-level cache across processor cores):

$$D[A[i] - \min(A)] \mathrel{+}= 1, \quad T_{\text{dense}}(n, \Delta_{\text{global}}) = O(n + \Delta_{\text{global}})$$

---

### Stage 4: Fused-Histogram Radix Sort & Degenerate-Byte Skipping (`radix_lsd_fused_sort`)

For 32-bit signed integers ($b = 4$ bytes) or 64-bit signed integers ($b = 8$ bytes), all $b$ histograms $H_0, \dots, H_{b-1} \in \mathbb{N}^{256}$ are accumulated in a single 4-way unrolled pass (`as_chunks::<4>()`):

$$H_r\left[ \left( (A[i] \gg (8r)) \oplus \delta_{r, b-1} \cdot \text{0x80} \right) \mathbin{\&} \text{0xff} \right] \mathrel{+}= 1 \quad \forall r \in \{0, \dots, b-1\}$$

where the Kronecker delta $\delta_{r, b-1}$ flips the sign bit (`0x80`) on the most significant byte so two's-complement negative integers precede positive integers.

**Degenerate-Byte Skipping**: Before executing scatter pass $r$, the kernel checks whether byte $r$ has zero Shannon entropy:

$$\exists c \in \{0, \dots, 255\} \text{ such that } H_r[c] = n \implies \text{Skip pass } r \text{ in } O(1)$$

If only $p \le b$ byte positions vary across the dataset, the total memory work drops from $2b \cdot n$ to $(1 + p) \cdot n$ array traversals.

---

### Stage 5: Hybrid MSB + L2-Local Radix Partitioning (`radix_msb_l2_hybrid_sort`)

When $n \ge 1{,}500{,}000$, scattering randomly across 256 buckets in a $4\text{ MB} \dots 40\text{ MB}$ array exceeds L2 capacity and induces TLB (Translation Lookaside Buffer, the hardware cache that translates virtual memory addresses to physical pages) thrashing. `latticesort` partitions the slice once by the signed MSB ($r = 3$) into 256 contiguous sub-problems $B_0, \dots, B_{255}$ of expected size:

$$\mathbb{E}[|B_c|] = \frac{n}{256} \quad (\approx 156\text{ KB at } n = 10{,}000{,}000)$$

Each bucket $B_c$ fits inside L2 cache, where the lower 24 bits ($r \in \{0, 1, 2\}$) are sorted locally via `l2_local_3pass_radix_sort` (or dispatched across CPU (Central Processing Unit, the primary processor executing program instructions) cores in `parallel_lattice_sort`).

---

### Stage 6: Order-Preserving Bijective Bit Projections (`LatticeSortable`)

To sort unsigned integers and IEEE (Institute of Electrical and Electronics Engineers, the standards body defining the 754 floating-point binary representation) 754 floating-point numbers using the signed integer kernels:

* **Unsigned Integers (`u32` / `u64`)**:

$$\phi_{\text{uint}}(u) = u \oplus 2^{w-1} \quad (w \in \{32, 64\})$$

* **IEEE 754 Floating-Point (`f32` / `f64`)**: In IEEE 754 sign-magnitude encoding, positive floats already ascend lexicographically with their bit pattern, whereas negative floats (`sign_bit = 1`) descend as magnitude increases. Defining the arithmetic sign-extension mask $M(bits) = \text{sar}(bits, w - 1) \mathbin{\&} (2^{w-1} - 1)$, the self-inverse bijection is:

$$\phi_{\text{float}}(bits) = bits \oplus \left( \text{sar}(bits, w - 1) \mathbin{\&} \text{0x7fff\_ffff} \right), \quad \phi_{\text{float}}^{-1} = \phi_{\text{float}}$$

This maps $[-\infty, -0.0, +0.0, +\infty]$ monotonically onto the signed integer lattice in $O(1)$ bitwise instructions without branches.

---

## Performance Benchmarks vs Literature Sorting Algorithms

### Benchmark Environment and Methodology
* **Hardware**: `13th Gen Intel(R) Core(TM) i7-1360P` (12 physical cores / 16 logical threads: 4 Performance-cores + 8 Efficiency-cores), with dynamic Turbo Boost enabled.
* **Cache Hierarchy**: L1d `48 KiB` per P-core (`32 KiB` per E-core), L2 `1.25 MiB` per P-core (`9 MiB` total across 6 instances), L3 `18 MiB` shared.
* **System & Compiler**: Linux x86_64, `rustc` (Edition 2024), `-C opt-level=3`, Link-Time Optimization (`lto = true`), `codegen-units = 1`.
* **Measurement Protocol & Thermal Variance**: Each benchmark computes the median latency across runs per distribution and verifies every output slice element-for-element against `std::slice::sort_unstable`. Across isolated binary runs (`latticesort`, `4.79x` to `4.99x` post Red-Team immunization in `exp13`..`exp19`) and sustained multi-algorithm sweeps (`bench_literature`, `5.68x`), the observed single-core geometric mean at $N = 1{,}000{,}000$ ranges between **4.79x and 5.68x** (`1.06 ms` to `1.15 ms` vs `5.22 ms` to `6.04 ms`) depending on P-core Turbo Boost residency.

### 1. Comparison Against 6 Sorting Algorithms from Computer Science Literature ($N = 1{,}000{,}000$)

Executed via `cargo run --release --bin bench_literature` across all 9 benchmark distributions:

| Literature Algorithm | Scientific Citation | Geometric Mean (9 Datasets) | Uniform 32-Bit Random | `latticesort` Geomean Speedup | `latticesort` Uniform32 Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **`latticesort`** | **This crate (`src/candidate.rs`)** | **`1.064 ms`** | **`5.206 ms`** | **1.00x (Baseline)** | **1.00x (Baseline)** |
| **1. `driftsort`** (`std::slice::sort`) | Bergdoll & Peters (2024), modern stable merge sort in Rust 1.81+ | `5.341 ms` | `19.840 ms` | **5.02x faster** | **3.81x faster** |
| **2. `ipnsort`** (`std::slice::sort_unstable`) | Bergdoll & Peters (2024), pattern-defeating network sort in Rust 1.81+ | `6.046 ms` | `16.465 ms` | **5.68x faster** | **3.16x faster** |
| **3. Classic 4-Pass LSD Radix Sort** | Seward (1954) / Knuth, *The Art of Computer Programming* Vol. 3 (1973) | `11.948 ms` | `6.470 ms` | **11.23x faster** | **1.24x faster** |
| **4. American Flag Sort (In-Place MSB Radix)** | McIlroy, Bostic & McIlroy (1993), 256-ary cyclic bucket permutation | `14.871 ms` | `26.658 ms` | **13.98x faster** | **5.12x faster** |
| **5. Bentley-McIlroy 3-Way Quicksort** | Bentley & McIlroy (1993), C `qsort` with DNF (Dutch National Flag, a three-way partitioning scheme dividing elements into less-than, equal-to, and greater-than regions) | `25.058 ms` | `75.871 ms` | **23.55x faster** | **14.57x faster** |
| **6. Floyd Bottom-Up HeapSort** | Williams & Floyd (1964), in-place binary heap sort | `75.216 ms` | `126.038 ms` | **70.70x faster** | **24.21x faster** |

---

### 2. Per-Distribution Breakdown ($N = 1{,}000{,}000$ `i32` Elements)

| Dataset Distribution | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` | Active Stage |
| :--- | ---: | ---: | ---: | :--- |
| 1. Cyclic `K=12` | `0.788` | `2.655` | **3.37x** | Dual-Sentinel L1 Hash Aggregation |
| 2. Cyclic `K=13` | `0.761` | `2.862` | **3.76x** | Dual-Sentinel L1 Hash Aggregation |
| 3. Cyclic `K=17` (prime) | `0.785` | `3.237` | **4.12x** | Dual-Sentinel L1 Hash Aggregation |
| 4. Wide `K=13` (span $10^9$) | `1.132` | `3.997` | **3.53x** | Dual-Sentinel L1 Hash Aggregation |
| 5. Random `K=100` | `0.858` | `6.143` | **7.16x** | Dual-Sentinel L1 Hash Aggregation |
| 6. Random `K=1000` | `1.204` | `9.826` | **8.16x** | Dense Range Counting (`span <= 1000`) |
| 7. Bitonic Soliton Wave ($\cap$) | `0.625` | `1.835` | **2.93x** | Bitonic Run Merger ($O(n)$) |
| 8. Pareto Power-Law | `1.148` | `12.660` | **11.03x** | Dense Range Counting |
| 9. Uniform 32-bit Random | `5.206` | `14.236` | **2.73x** | 4-Pass Fused LSD Radix Sort |
| **Geometric Mean ($N = 1\text{M}$)** | **`1.113`** | **`5.547`** | **4.98x** | **Full Adaptive Pipeline** |

---

### 3. Large-Scale Benchmark Suite ($N = 10{,}000{,}000$ `i32` Elements)

| Dataset Distribution ($N = 10\text{M}$) | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` | Active Stage |
| :--- | ---: | ---: | ---: | :--- |
| 1. Cyclic `K=12` | `9.05` | `28.59` | **3.16x** | Dual-Sentinel L1 Hash Aggregation |
| 2. Cyclic `K=13` | `9.18` | `28.91` | **3.15x** | Dual-Sentinel L1 Hash Aggregation |
| 3. Cyclic `K=17` (prime) | `8.98` | `32.49` | **3.62x** | Dual-Sentinel L1 Hash Aggregation |
| 4. Wide `K=13` (span $10^9$) | `14.12` | `40.48` | **2.87x** | Dual-Sentinel L1 Hash Aggregation |
| 5. Random `K=100` | `13.45` | `61.49` | **4.57x** | Dual-Sentinel L1 Hash Aggregation |
| 6. Random `K=1000` | `14.28` | `97.69` | **6.84x** | Dense Range Counting |
| 7. Bitonic Soliton Wave ($\cap$) | `9.96` | `22.74` | **2.28x** | Bitonic Run Merger ($O(n)$) |
| 8. Pareto Power-Law | `32.63` | `149.49` | **4.58x** | Dense Range Counting |
| 9. Uniform 32-bit Random | `64.35` | `179.96` | **2.80x** | Hybrid MSB + L2-Local 3-Pass Radix |
| **Geometric Mean ($N = 10\text{M}$)** | **`15.40`** | **`61.93`** | **4.02x** | **Full Adaptive Pipeline** |

---

### 4. Polymorphic Types (`LatticeSortExt`) at $N = 1{,}000{,}000$

| Type and Distribution ($N = 1\text{M}$) | `latticesort` (ms) | `std::sort_unstable` (ms) | Speedup vs `std` |
| :--- | ---: | ---: | ---: |
| 1. `u32` (Uniform 32-bit) | `7.10` | `14.26` | **2.01x** |
| 2. `i64` (Timestamps: 4 constant upper bytes skipped) | `12.40` | `14.84` | **1.20x** |
| 3. `i64` (Low-Cardinality `K=50` 64-bit keys) | `1.68` | `4.55` | **2.70x** |
| 4. `u64` (Bounded 32-bit keys in `u64`) | `14.18` | `15.36` | **1.08x** |
| 5. `f32` (Signed IEEE 754 floats `+/-`) | `6.01` | `24.52` | **4.08x** |
| 6. `f64` (Sensor readings `K=64` `+/-`) | `2.77` | `5.11` | **1.85x** |

---

### 5. Multi-Core Parallel Execution (`parallel::parallel_lattice_sort`) and 1-Billion-Element Scale ($4.00\text{ GB}$)

| Uniform 32-Bit Scenario | `parallel_lattice_sort` | Single-Thread `latticesort` | `std::sort_unstable` (`ipnsort`) | Parallel Speedup vs `std` | Single-Thread Speedup vs `std` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| $N = 1{,}000{,}000$ ($4\text{ MB}$) | `3.04 ms` | `5.41 ms` | `14.52 ms` | **4.77x** | **2.68x** |
| $N = 10{,}000{,}000$ ($40\text{ MB}$) | `32.03 ms` | `64.87 ms` | `161.84 ms` | **5.05x** | **2.49x** |
| **$N = 1{,}000{,}000{,}000$ ($4.00\text{ GB}$)** | **`2.913 s` (`343.2M/s`)** | **`7.966 s` (`125.5M/s`)** | **`22.371 s` (`44.7M/s`)** | **7.68x** | **2.81x** |

---

### 6. Red-Team Adversarial Falsification Suite ($N = 1{,}000{,}000$ and $N = 10{,}000{,}000$)

| Adversarial Attack Vector | $N = 1\text{M}$ Speedup vs `std` | $N = 10\text{M}$ Speedup vs `std` | Verdict |
| :--- | ---: | ---: | :--- |
| 1A. Iceberg Attack (95% constant `0` + 5% random tail) | **6.10x** | **4.02x** | Survived |
| 1B. Mid-Array Entropy Trap (200 keys hidden at 45%..65%) | **3.10x** | **1.92x** | Survived |
| 1C. Sentinel & 24-Way Hash Collision Torture | **1.72x** | **1.04x** | Survived |
| 2A. Cutoff Boundary Uniform Random ($N = 600$ / $N = 6{,}000$) | **1.41x** | **2.21x** | Survived |
| 2B. Mid-Point Single Swap (`swap(n/2 - 2, n/2 - 1)`) | **1.14x** | **1.54x** | Survived |
| 3A. Fake Soliton (92% ascending ramp + 8% oscillating tail) | **2.86x** | **2.58x** | Survived |
| 3B. Valley Soliton ($\cup$-shape descending then ascending) | **3.51x** | **2.78x** | Survived |
| 4A. Sparse Horizon (`K=150` keys, span $10^6$) | **4.43x** | **1.96x** | Survived |
| 4B. Exact Golden-Stride Probe Spoof (planted zeros at stride) | **2.68x** | **2.75x** | Survived |

---

## Quick Start

```rust
use latticesort::{LatticeSortExt, parallel};

fn main() {
    // 1. Polymorphic slice sorting via LatticeSortExt (i32, u32, i64, u64, f32, f64)
    let mut timestamps: Vec<i64> = vec![1_760_000_005_000, 1_760_000_001_000, 1_760_000_003_000];
    timestamps.lattice_sort();
    assert!(timestamps.is_sorted());

    let mut readings: Vec<f64> = vec![3.5, -0.0, 0.0, -12.8, 1.25];
    readings.lattice_sort();

    // 2. Multi-core parallel sorting for large i32 batches
    let mut batch: Vec<i32> = (0..1_000_000).rev().collect();
    parallel::parallel_lattice_sort(&mut batch);
    assert!(batch.is_sorted());
}
```

---

## Reproducing the Benchmarks via CLI (Command-Line Interface, text-based terminal interface for running programs)

```bash
# Primary 1,000,000-element benchmark suite (9 distributions)
cargo run --release --bin latticesort

# Comparison against 6 sorting algorithms from Computer Science literature
cargo run --release --bin bench_literature

# Red-team adversarial falsification suite (1,000,000 elements)
cargo run --release --bin adversarial

# 10x scale benchmark + red-team suite (10,000,000 elements)
cargo run --release --bin bench_10m

# Polymorphic types (u32, i64, u64, f32, f64) and multi-core parallel benchmark
cargo run --release --bin bench_fronts_3_4

# 1-Billion-element (4.00 GB) uniform 32-bit random benchmark
cargo run --release --bin bench_1b
```

---

## Interactive 3D Visualization (Three.js r187)

An interactive 3D WebGL visualizer of the internal stages of `latticesort` is included in `viz/index.html`. Serve the repository root with any local static server to explore the 3D scene in your browser.

---

## License

Released under the [MIT (Massachusetts Institute of Technology, permissive free software license) License](https://github.com/SimpleSoftwareLTDA/latticesort/blob/main/LICENSE). Copyright (c) 2026 Robson Cassiano.

---

## Citing

Author: [Robson Cassiano](https://orcid.org/0009-0009-5188-361X) (ORCID (Open Researcher and Contributor ID, persistent digital identifier for researchers): [0009-0009-5188-361X](https://orcid.org/0009-0009-5188-361X)).

For the source code, cite the software entry below. GitHub's "Cite this repository" button reads [CITATION.cff](https://github.com/SimpleSoftwareLTDA/latticesort/blob/main/CITATION.cff).

```bibtex
@software{cassiano2026latticesort,
  author = {Cassiano, Robson},
  title = {{latticesort}: adaptive structural, hash-aggregated, and cache-aware hybrid radix sorting in safe Rust},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/SimpleSoftwareLTDA/latticesort},
  license = {MIT},
  note = {Author ORCID: https://orcid.org/0009-0009-5188-361X}
}
```
