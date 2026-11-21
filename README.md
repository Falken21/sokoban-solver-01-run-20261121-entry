# Sokoban Solver — make.fun sokoban-solver-01-run-20261121

A push-space A* Sokoban solver in a single dependency-free `main.ts`
(Node 22, type-stripping). It answers all 5 instances of a set in one run.

## How it works

The generator guarantees a 1-wide hallway with 1–4 rooms, each room holding
3–4 boxes and as many goals, and a box pushed out of its room can never
return. The solver therefore:

1. splits the level into independent rooms (flood fill off the hallway row);
2. solves each room push-optimally with A* over push states (sorted box
   positions + player region), using an admissible optimal-assignment
   heuristic (Hungarian/brute-force over BFS distances), static dead-square
   pruning (reverse push-graph reachability from the goals), a wall-only
   frozen-box prune, and a transposition table;
3. stitches the room solutions together with hallway walks, then replays
   the whole move string on the full level to verify it before writing it.

All search budgets are expansion counts, never wall-clock, so two runs
produce byte-identical output. A greedy best-first fallback covers a room
whose A* exhausts its budget; an unsolved room yields an empty solution
for that level rather than an unverified move string.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Practice results (kit practice set)

| instance | rooms | P* | pushes | moves | expansions |
|----------|-------|----|--------|-------|------------|
| i001     | 1     | 13 | 13     | 46    | 29         |
| i002     | 1     | 16 | 16     | 77    | 117        |
| i003     | 2     | 29 | 29     | 108   | 29         |
| i004     | 3     | 47 | 47     | 209   | 796        |
| i005     | 4     | 60 | 60     | 299   | 1562       |

Kit scorer: **1,000,000** (all 5 instances solved in exactly P* pushes).
Whole set runs in ~0.6 s; two runs are byte-identical.
