# Glossary

Plain-English definitions of the jargon used across this project. Each entry says what
the term means and how it's used here. If a doc or comment confuses you, start here.

> New to the whole topic? Read these four first: **vector**, **embedding**,
> **nearest neighbour search**, **index**. The rest builds on them.

---

## The big picture

**Vector**
: An ordered list of numbers, e.g. `[0.12, -0.7, 0.33, ...]`. In this project a vector
  has a fixed length called its **dimension** (e.g. 128 numbers). Think of it as a point
  in space — a 128-dimensional space, in that example.

**Embedding**
: A vector produced by a model to represent the *meaning* of something (a sentence, an
  image). The key property: things with similar meaning get vectors that are close
  together. "dog" and "puppy" end up near each other; "dog" and "Tuesday" do not.

**Dimension (dim)**
: How many numbers are in each vector. All vectors in one database share the same
  dimension. Higher dimension = more detail but more memory and slower math.
  **Fixed per `Engine`:** one `Engine` holds one dimension (like a Qdrant *collection* or a
  FAISS *index*). For two datasets of different dimensions, create two `Engine`s — you can't
  mix dimensions, because distance is only defined between vectors of equal length. Even at
  equal dimension, don't mix embeddings from different models: their coordinate systems
  aren't comparable.

**Nearest neighbour search**
: The core operation. Given a query vector, find the stored vectors *closest* to it.
  "Closest" is measured by a **distance metric** (below).

**k-NN / top-k**
: Return the `k` closest vectors (e.g. top-10), not just the single closest.

**Vector database**
: A system that stores many vectors and answers nearest-neighbour queries quickly. This
  project is a small one. Real examples: FAISS, Qdrant, Milvus.

**Index**
: A data structure that makes search fast. Without one you'd compare the query to every
  stored vector (slow). An index organizes vectors so you can skip most of them. This
  project has two: **Flat** (no shortcut, exact) and **HNSW** (a smart shortcut). An index
  knows nothing about disk, deletion rules, threads, or users — only "given a query, return
  the nearest ids." `index.hpp` is the interface; `flat_index.hpp` / `hnsw_index.hpp` are
  the implementations.

**Engine**
: The top-level object you actually use, and the orchestrator around the index. It owns the
  vector store + one index + the write-ahead log, and coordinates everything that *isn't*
  the raw search algorithm: insert/update/delete (logging changes for durability), applying
  metadata filters, running queries across threads, and crash recovery. For search it
  **delegates** the nearest-neighbour work to its index. Named like a "database engine" /
  "storage engine." Files: `engine.hpp` / `engine.cpp`.
: *Index vs Engine, in one line:* the **index** finds nearest neighbours; the **engine** is
  the database around it (storage, durability, filtering, concurrency) that calls the index.
  Swapping Flat for HNSW changes the index, not the engine.

---

## Memory and data layout

**Arena**
: One big pre-allocated array that holds *all* the vectors back-to-back, instead of
  allocating each vector separately. Vector `i` lives at `arena[i*dim .. (i+1)*dim)`.
  Fast to append to, and fast to scan because the data is contiguous. (See the README/chat
  for the binder analogy.)

**SoA (Structure of Arrays)**
: A layout where related data is kept in parallel arrays rather than one array of objects.
  Here it just means "the raw floats live together in the arena," which is cache-friendly.
  The opposite, AoS (Array of Structs), would scatter each vector inside its own object.

**Cache locality**
: The CPU fetches memory in small blocks ("cache lines", ~64 bytes) and is *much* faster
  when the data it needs next is already in that block. Packing vectors contiguously
  (the arena) gives good cache locality, so scanning is fast.

**SIMD (Single Instruction, Multiple Data)**
: A CPU feature that does the same math on several numbers at once (e.g. multiply 8 floats
  in one instruction). Great for distance math. Not used yet here (Apple Silicon would need
  NEON), but the contiguous layout is what makes it possible later.

**Dense id**
: Ids that are tightly packed integers `0, 1, 2, 3, ...` with no gaps. They double as array
  indices, so "id 5" directly means "slot 5 in the arena."

**Internal id vs external id**
: The *external id* is what you (the user) call a vector — a string like `"doc-42"`. The
  *internal id* is the dense integer the system assigns it. A hash map translates between
  them. Indexes only ever store the small internal ids.

**Tombstone**
: A "deleted" marker instead of actually removing data. Deleting flips a flag (`live = 0`)
  rather than shifting the whole arena — O(1) instead of O(N). The slot is skipped in
  results and can be reused later.

---

## Distances (how "closeness" is measured)

**Distance metric**
: A formula for how far apart two vectors are. This project uses the convention
  **smaller = closer** for all of them, so sorting is always "smallest first."

**L2 / Euclidean distance**
: Straight-line distance, like a ruler between two points. The code uses *squared* L2
  (skips the square root) because it sorts the same way and is faster.

**Cosine similarity**
: Measures the *angle* between two vectors, ignoring their length. Common for text
  embeddings. The code stores it as `1 - cosine_similarity` so that 0 = identical direction
  and bigger = farther (keeping "smaller = closer").

**Dot product**
: Multiply matching elements and sum them. The code negates it (`-dot`) so larger
  similarity becomes a smaller distance.

**Normalize / L2 normalize**
: Scale a vector so its length is exactly 1. After normalizing, dot product equals cosine
  similarity. The mock embedding does this.

---

## Indexes and HNSW

**Flat index**
: The simplest index: compare the query against every vector. Always exact, but O(N) per
  query. Used here as the **oracle** — the source of truth for checking HNSW's quality.

**Oracle**
: A known-correct reference you compare against. Flat is exact, so HNSW is "good" if it
  returns nearly the same results as Flat.

**ANN (Approximate Nearest Neighbour)**
: Search that's allowed to be *slightly* wrong in exchange for being much faster. HNSW is
  an ANN method. You trade a little accuracy (**recall**) for a lot of speed.

**HNSW (Hierarchical Navigable Small World)**
: The ANN algorithm used here. It builds a **graph** (vectors connected to their near
  neighbours) with multiple **layers**. Search starts at the top (few nodes, big jumps) and
  zooms down to the bottom (all nodes, fine steps), following links toward the query.

**Graph / node / edge / neighbour**
: A graph is dots (**nodes**) connected by lines (**edges**). Here each node is a vector,
  and its edges link it to nearby vectors (its **neighbours**). Search walks these links.

**Layer**
: HNSW stacks several graphs. Upper layers have few nodes and act like an express highway
  for covering distance quickly; the bottom layer has every node for precise results.

**Entry point**
: The single node where every search begins (top of the top layer).

**Greedy search**
: "Always step to whichever neighbour is closest to the query, repeat until none is closer."
  Simple and fast; used to descend the upper layers.

**M**
: HNSW tuning knob: how many neighbour links each node keeps per layer (default 16; the
  bottom layer allows 2×M). More links = better recall but more memory and slower build.

**efConstruction / efSearch (ef = "exploration factor")**
: How many candidate nodes to keep "in play" while exploring. `efConstruction` is used when
  *building* the graph; `efSearch` when *querying*. Higher = better quality, slower. The
  recall-vs-latency curve is produced by sweeping `efSearch`.

**Recall@k**
: The accuracy measure for ANN. Of the true top-k nearest (from Flat), what fraction did
  the index actually return? `recall@10 = 0.95` means it found 9.5 of the true 10 on
  average. 1.0 = perfect.

---

## Performance vocabulary

**Latency**
: How long one query takes. Reported in microseconds (µs).

**p50 / p95 / p99 (percentiles)**
: p50 = the median (half of queries are faster). p95 = 95% of queries are faster than this;
  it captures the slow tail. Tail latency matters more than the average, which is why we
  report percentiles, not the mean.

**QPS (Queries Per Second) / throughput**
: How many queries the system handles per second. Goes up with more threads.

**Big-O (O(1), O(N), O(N·dim))**
: Shorthand for how cost grows with size. O(1) = constant (doesn't grow). O(N) = grows
  linearly with the number of items. Flat search is O(N·dim) per query; a tombstone delete
  is O(1).

---

## Durability (surviving a crash)

**Persistence / durability**
: Keeping data safe so it survives the program stopping or the machine crashing — i.e.
  writing it to disk, not just RAM.

**WAL (Write-Ahead Log)**
: An append-only file where every change is recorded *before* it's applied in memory ("log
  first, then act"). After a crash you replay the log to rebuild state. This is how real
  databases stay durable.

**fsync**
: A command that forces the operating system to actually flush data to physical disk
  (normally it buffers writes for speed). Without fsync, a crash can lose "written" data.

**Checksum / CRC (Cyclic Redundancy Check)**
: A small number computed from a chunk of data so you can detect if it got corrupted. If
  the stored CRC doesn't match the recomputed one, the data is bad.

**Torn write**
: A write that was interrupted mid-way (e.g. crash during append), leaving a half-written
  record. Recovery detects this (short read or bad CRC) and stops there, keeping everything
  before it.

**Snapshot**
: A saved copy of the full current state at a point in time. Combined with a WAL: load the
  snapshot, then replay only the log entries that came after it. Lets the log stay small.

**Op / operation**
: One logical change: Insert, Update, or Delete. The whole system is driven by a stream of
  ops.

**Event sourcing / derived (materialized) view**
: A design where the log of ops *is* the source of truth, and everything else (the store,
  the index) is rebuilt by replaying that log. The index here is a "derived view" — it's
  not saved separately; it's reconstructed from the op log on startup.

---

## Concurrency (doing things in parallel)

**Thread**
: An independent line of execution. Multiple threads run at the same time on multiple CPU
  cores. Running queries on several threads = more QPS.

**Thread pool / chunking**
: Splitting work across threads. Here the batch of queries is divided into contiguous
  chunks, one per worker thread.

**`std::jthread`**
: C++'s "joining thread" — a thread that automatically waits for itself to finish when it
  goes out of scope. Less error-prone than the older `std::thread`.

**Race condition**
: A bug where two threads touch the same data at the same time and the result depends on
  timing. Avoided here by making the search path read-only (no shared data is modified).

**Immutable / read-only path**
: Code that only *reads* shared data, never changes it. Many threads can safely read the
  same thing at once. Search is read-only, which is why parallel search is safe.

**Single-writer**
: The rule that only one thread changes data at a time (while many may read). It sidesteps
  the hardest concurrency bugs.

**happens-before / join**
: "Join" means waiting for a thread to finish. We join all workers *before* reading their
  results, which guarantees their writes are visible (the writes "happen before" the read).

---

## RAG (Retrieval-Augmented Generation)

**RAG**
: A technique where, before asking a language model a question, you *retrieve* relevant
  documents from a database and stuff them into the prompt as context — so the model
  answers using real, grounded information instead of guessing.

**Token**
: Roughly a word or word-piece. Models and budgets are measured in tokens. This project
  approximates tokens as whitespace-separated words.

**Chunking / sliding window / overlap**
: Splitting a long document into smaller pieces ("chunks") so each fits a model and can be
  embedded. A *sliding window* moves along the text in fixed-size steps; *overlap* means
  consecutive chunks share some tokens so meaning isn't cut off at the boundary.

**Token budget**
: The maximum number of tokens the final prompt may use. The prompt builder packs in the
  most relevant chunks until the budget is reached, reserving room for the question.

**Feature hashing**
: A cheap trick to turn words into vector positions: hash each word to an index and bump
  that slot. The mock embedding uses this (it is *not* a real semantic model — just enough
  to exercise the pipeline offline).

**Prompt**
: The text sent to a language model. Here it's "instructions + retrieved context +
  question," assembled by the `PromptBuilder`.

---

## C++ / software terms

**Index (the C++ class) vs index (the search structure)**
: Confusingly, "index" means both "the thing that speeds up search" (Flat/HNSW) and, in
  arrays, "a position." Context tells them apart.

**`std::span`**
: A lightweight "view" of a contiguous range (pointer + length) without owning or copying
  it. Used to pass vectors around cheaply.

**Template / compile-time policy**
: A C++ way to write code once that works for many types, resolved at compile time. The
  distance metrics are templates, so the math gets baked directly into the search loop
  (fast — no indirection).

**Concept**
: A named set of requirements a template type must satisfy (e.g. "has a `distance`
  function"). Gives clear errors and documents intent.

**Virtual dispatch / polymorphism**
: Choosing which function to call *at runtime* through a base-class pointer (here, the
  `Index` interface). Flexible but slightly slower — so it's used only at the once-per-query
  boundary, never in the tight inner loop.

**Inlining**
: The compiler pasting a small function's body directly into the caller, removing call
  overhead. Compile-time policies (templates) inline; virtual calls do not.

**Factory**
: A function that builds and returns an object, often choosing the concrete type at runtime
  (e.g. `make_hnsw_index` picks the right metric template).

**Facade**
: A class that wraps several pieces behind one simple interface. `Engine` is the facade over
  the store + index + WAL.

**Seam**
: A deliberate boundary/interface where you can swap an implementation later without
  changing surrounding code (e.g. `LogStore` lets the WAL write to a file now, something
  else later).

**Bitset / allowed-set**
: A compact array of on/off bits, one per id. The metadata filter produces an allowed-set
  bitset: bit `i` is on if vector `i` passed the filter and may be returned.

**Heap / priority queue (min-heap / max-heap)**
: A structure that always gives quick access to the smallest (min-heap) or largest
  (max-heap) item. Top-k search keeps a size-k max-heap of the best results so it can cheaply
  drop the worst when a better candidate arrives.
