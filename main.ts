// Sokoban solver — push-space A* with per-room decomposition.
//
// Strategy (see README.md): the generator guarantees a 1-wide hallway with
// 1-4 rooms, each room holding 3-4 boxes and as many goals, and a box pushed
// out of its room can never return. So every room is solved independently
// (push-optimal A* from its door) and the room solutions are stitched
// together with hallway walks. Cost is counted in PUSHES, because the
// score is computed from pushes. All budgets are expansion counts, never
// wall-clock, so two runs produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets only (determinism: no clock is ever read).
const ROOM_ASTAR_CAP = 300000;
const ROOM_GREEDY_CAP = 150000;
const WHOLE_ASTAR_CAP = 400000;
const WHOLE_GREEDY_CAP = 200000;
// Safety net for the 60 s wall budget of the whole set: at the measured
// ~8,000 expansions/s, 250,000 expansions cost ~31 s worst case. The
// practice set needs only ~2,500, so this never binds on real levels.
const GLOBAL_EXPANSION_CAP = 250000;
let globalExpansions = 0;

const DIRS = [
  { dr: -1, dc: 0, ch: 'u' },
  { dr: 1, dc: 0, ch: 'd' },
  { dr: 0, dc: -1, ch: 'l' },
  { dr: 0, dc: 1, ch: 'r' },
] as const;

type Level = {
  w: number;
  h: number;
  wall: boolean[];
  goals: Set<number>;
  boxes: number[];
  player: number;
  floor: number[];
};

type Room = {
  cells: Set<number>;
  door: number;
  entry: number;
  boxes: number[];
  goals: number[];
};

type SearchStats = { expansions: number; fallback: boolean };

const idx = (r: number, c: number, w: number): number => r * w + c;
const rowOf = (i: number, w: number): number => Math.floor(i / w);
const colOf = (i: number, w: number): number => i % w;

function parseLevel(text: string): Level {
  const rawLines = text.replace(/\r/g, '').split('\n');
  while (rawLines.length > 0 && rawLines[rawLines.length - 1] === '') rawLines.pop();
  const h = rawLines.length;
  const w = Math.max(...rawLines.map((l) => l.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = -1;
  const floor: number[] = [];
  for (let r = 0; r < h; r++) {
    const line = rawLines[r].padEnd(w, '#');
    for (let c = 0; c < w; c++) {
      const ch = line[c];
      const i = idx(r, c, w);
      if (ch === '#') continue;
      wall[i] = false;
      floor.push(i);
      if (ch === '.' || ch === '*' || ch === '+') goals.add(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player, floor };
}

/** Connected components of floor cells, restricted to rows != hallwayRow. */
function detectRooms(lv: Level): { rooms: Room[]; stray: number[] } {
  const hallwayRow = rowOf(lv.player, lv.w);
  const seen = new Set<number>();
  const rooms: Room[] = [];
  const inRoom = new Set<number>();
  const sortedFloor = [...lv.floor].sort((a, b) => a - b);
  for (const start of sortedFloor) {
    if (rowOf(start, lv.w) === hallwayRow || seen.has(start)) continue;
    // Flood fill this component (never crossing the hallway row).
    const comp: number[] = [];
    const stack = [start];
    seen.add(start);
    while (stack.length > 0) {
      const cur = stack.pop() as number;
      comp.push(cur);
      for (const d of DIRS) {
        const nr = rowOf(cur, lv.w) + d.dr;
        const nc = colOf(cur, lv.w) + d.dc;
        if (nr < 0 || nr >= lv.h || nc < 0 || nc >= lv.w) continue;
        const ni = idx(nr, nc, lv.w);
        if (lv.wall[ni] || rowOf(ni, lv.w) === hallwayRow || seen.has(ni)) continue;
        seen.add(ni);
        stack.push(ni);
      }
    }
    comp.sort((a, b) => a - b);
    // Door: a component cell adjacent to a hallway-row floor cell.
    let door = -1;
    let entry = -1;
    for (const cell of comp) {
      for (const d of DIRS) {
        const nr = rowOf(cell, lv.w) + d.dr;
        const nc = colOf(cell, lv.w) + d.dc;
        if (nr !== hallwayRow || nc < 0 || nc >= lv.w) continue;
        const ni = idx(nr, nc, lv.w);
        if (!lv.wall[ni]) {
          door = cell;
          entry = ni;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) continue; // component not attached to the hallway; ignore.
    const cells = new Set<number>(comp);
    const boxes = lv.boxes.filter((b) => cells.has(b));
    const goals = [...lv.goals].filter((g) => cells.has(g)).sort((a, b) => a - b);
    for (const c of comp) inRoom.add(c);
    rooms.push({ cells, door, entry, boxes, goals });
  }
  const stray = lv.boxes.filter((b) => !inRoom.has(b));
  return { rooms, stray };
}

// ---------------------------------------------------------------------------
// Push-space A* over one board (a room sub-board, or the whole level).
// floor: the cells the player/boxes may occupy. forbidden: cells a box may
// never be pushed onto (the hallway cell outside a room door).
// ---------------------------------------------------------------------------

type SearchNode = {
  key: string;
  boxes: number[];
  player: number;
  g: number;
  h: number;
  parent: string | null;
  move: string;
};

type HeapItem = { f: number; h: number; seq: number; key: string };

function heapPush(heap: HeapItem[], item: HeapItem): void {
  heap.push(item);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    const a = heap[i];
    const b = heap[p];
    if (a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.seq < b.seq)))) {
      heap[i] = b;
      heap[p] = a;
      i = p;
    } else break;
  }
}

function heapPop(heap: HeapItem[]): HeapItem {
  const top = heap[0];
  const last = heap.pop() as HeapItem;
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let m = i;
      const better = (x: HeapItem, y: HeapItem): boolean =>
        x.f < y.f || (x.f === y.f && (x.h < y.h || (x.h === y.h && x.seq < y.seq)));
      if (l < heap.length && better(heap[l], heap[m])) m = l;
      if (r < heap.length && better(heap[r], heap[m])) m = r;
      if (m === i) break;
      const t = heap[i];
      heap[i] = heap[m];
      heap[m] = t;
      i = m;
    }
  }
  return top;
}

/** BFS distances between all floor cells, ignoring boxes. */
function allPairsDist(floor: number[], wall: boolean[], w: number, h: number): Map<string, number> {
  const floorSet = new Set<number>(floor);
  const dist = new Map<string, number>();
  for (const s of floor) {
    const d = new Map<number, number>();
    d.set(s, 0);
    const q = [s];
    let qi = 0;
    while (qi < q.length) {
      const cur = q[qi++];
      const cd = d.get(cur) as number;
      for (const dir of DIRS) {
        const nr = rowOf(cur, w) + dir.dr;
        const nc = colOf(cur, w) + dir.dc;
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const ni = idx(nr, nc, w);
        if (!floorSet.has(ni) || d.has(ni)) continue;
        d.set(ni, cd + 1);
        q.push(ni);
      }
    }
    for (const t of floor) dist.set(s + ',' + t, d.has(t) ? (d.get(t) as number) : 1e9);
  }
  return dist;
}

/**
 * Static dead squares: floor cells (not goals) from which no goal is
 * reachable by pushes. Reverse BFS on the push graph from the goals:
 * a box at x can be pushed to adjacent y iff x-d is floor, so the
 * predecessors of y are x = y-d with y-2d floor. Covers corners and
 * goal-less wall segments.
 */
function deadSquares(floor: number[], goals: Set<number>, wall: boolean[], w: number, h: number): Set<number> {
  const floorSet = new Set<number>(floor);
  const live = new Set<number>(goals);
  const q = [...goals];
  let qi = 0;
  while (qi < q.length) {
    const y = q[qi++];
    for (const d of DIRS) {
      const xr = rowOf(y, w) - d.dr;
      const xc = colOf(y, w) - d.dc;
      const br = rowOf(y, w) - 2 * d.dr;
      const bc = colOf(y, w) - 2 * d.dc;
      if (xr < 0 || xr >= h || xc < 0 || xc >= w) continue;
      if (br < 0 || br >= h || bc < 0 || bc >= w) continue;
      const x = idx(xr, xc, w);
      const behind = idx(br, bc, w);
      if (!floorSet.has(x) || !floorSet.has(behind) || live.has(x)) continue;
      live.add(x);
      q.push(x);
    }
  }
  const dead = new Set<number>();
  for (const f of floor) if (!live.has(f)) dead.add(f);
  return dead;
}

/** Shortest walk (u/d/l/r) from `from` to `to` avoiding walls and boxes. */
function walkPath(
  from: number,
  to: number,
  floorSet: Set<number>,
  boxSet: Set<number>,
  w: number,
  h: number,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, { p: number; ch: string }>();
  const q = [from];
  prev.set(from, { p: -1, ch: '' });
  let qi = 0;
  while (qi < q.length) {
    const cur = q[qi++];
    for (const d of DIRS) {
      const nr = rowOf(cur, w) + d.dr;
      const nc = colOf(cur, w) + d.dc;
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const ni = idx(nr, nc, w);
      if (!floorSet.has(ni) || boxSet.has(ni) || prev.has(ni)) continue;
      prev.set(ni, { p: cur, ch: d.ch });
      if (ni === to) {
        let s = '';
        let n = ni;
        while (n !== from) {
          const e = prev.get(n) as { p: number; ch: string };
          s = e.ch + s;
          n = e.p;
        }
        return s;
      }
      q.push(ni);
    }
  }
  return null;
}

/** All permutations of an array (n <= 4 in practice). */
function permutations<T>(arr: T[]): T[][] {
  if (arr.length <= 1) return [arr.slice()];
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permutations(rest)) out.push([arr[i], ...p]);
  }
  return out;
}

function astarSearch(
  floor: number[],
  goals: Set<number>,
  boxes0: number[],
  player0: number,
  forbidden: Set<number>,
  cap: number,
  greedy: boolean,
  w: number,
  h: number,
  wall: boolean[],
  stats: SearchStats,
): { moves: string; pushes: number } | null {
  const floorSet = new Set<number>(floor);
  const goalList = [...goals].sort((a, b) => a - b);
  const dist = allPairsDist(floor, wall, w, h);
  const dead = deadSquares(floor, goals, wall, w, h);
  const goalPerms = goalList.length === boxes0.length && goalList.length <= 4 ? permutations(goalList) : null;
  // Heuristic results repeat across states that share a box set but differ
  // in player region; memoise on the sorted box list.
  const hCache = new Map<string, number>();

  const heuristic = (boxes: number[]): number => {
    const ck = boxes.join(',');
    const cached = hCache.get(ck);
    if (cached !== undefined) return cached;
    let value: number;
    if (goalPerms !== null) {
      let best = 1e9;
      for (const perm of goalPerms) {
        let s = 0;
        for (let i = 0; i < boxes.length; i++) s += dist.get(boxes[i] + ',' + perm[i]) as number;
        if (s < best) best = s;
      }
      value = best;
    } else {
      let s = 0;
      for (const b of boxes) {
        let m = 1e9;
        for (const g of goalList) {
          const d = dist.get(b + ',' + g) as number;
          if (d < m) m = d;
        }
        s += m;
      }
      value = s;
    }
    hCache.set(ck, value);
    return value;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goals.has(b));

  // Region of the player: flood fill over floor minus boxes; its canonical
  // id is the smallest cell in it (deterministic).
  const regionOf = (player: number, boxSet: Set<number>): { set: Set<number>; min: number } => {
    const set = new Set<number>();
    const q = [player];
    set.add(player);
    let qi = 0;
    let min = player;
    while (qi < q.length) {
      const cur = q[qi++];
      if (cur < min) min = cur;
      for (const d of DIRS) {
        const nr = rowOf(cur, w) + d.dr;
        const nc = colOf(cur, w) + d.dc;
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const ni = idx(nr, nc, w);
        if (!floorSet.has(ni) || boxSet.has(ni) || set.has(ni)) continue;
        set.add(ni);
        q.push(ni);
      }
    }
    return { set, min };
  };

  // A box is frozen only if NO push direction is possible even with all
  // other boxes removed: a box blocked merely by another box is NOT frozen,
  // because the other box can move away first. Judging against walls (and
  // forbidden cells) alone keeps this prune sound; anything subtler is left
  // to the search, which handles dead ends naturally.
  const frozen = (boxes: number[]): boolean => {
    for (const b of boxes) {
      if (goals.has(b)) continue;
      let movable = false;
      for (const d of DIRS) {
        const tr = rowOf(b, w) + d.dr;
        const tc = colOf(b, w) + d.dc;
        const br = rowOf(b, w) - d.dr;
        const bc = colOf(b, w) - d.dc;
        if (tr < 0 || tr >= h || tc < 0 || tc >= w || br < 0 || br >= h || bc < 0 || bc >= w) continue;
        const t = idx(tr, tc, w);
        const behind = idx(br, bc, w);
        if (floorSet.has(t) && !forbidden.has(t) && floorSet.has(behind) && !forbidden.has(behind)) {
          movable = true;
          break;
        }
      }
      if (!movable) return true;
    }
    return false;
  };

  const keyOf = (boxes: number[], regionMin: number): string => boxes.join(',') + '|' + regionMin;

  const startBoxes = [...boxes0].sort((a, b) => a - b);
  if (startBoxes.some((b) => dead.has(b) || forbidden.has(b))) return null;
  const startRegion = regionOf(player0, new Set<number>(startBoxes));
  const startKey = keyOf(startBoxes, startRegion.min);
  const h0 = heuristic(startBoxes);
  const nodes = new Map<string, SearchNode>();
  nodes.set(startKey, { key: startKey, boxes: startBoxes, player: player0, g: 0, h: h0, parent: null, move: '' });
  const heap: HeapItem[] = [];
  let seq = 0;
  heapPush(heap, { f: greedy ? h0 : h0, h: h0, seq: seq++, key: startKey });

  while (heap.length > 0) {
    const item = heapPop(heap);
    const node = nodes.get(item.key) as SearchNode;
    if (node.g > 0 && item.f !== (greedy ? node.h : node.g + node.h)) continue; // stale entry
    if (solved(node.boxes)) {
      // Reconstruct: one recorded transition per push.
      let moves = '';
      let cur: SearchNode | undefined = node;
      const parts: string[] = [];
      while (cur && cur.parent !== null) {
        parts.push(cur.move);
        cur = nodes.get(cur.parent);
      }
      parts.reverse();
      moves = parts.join('');
      return { moves, pushes: parts.length };
    }
    stats.expansions++;
    globalExpansions++;
    if (stats.expansions > cap || globalExpansions > GLOBAL_EXPANSION_CAP) return null;
    const boxSet = new Set<number>(node.boxes);
    const region = regionOf(node.player, boxSet);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      for (const d of DIRS) {
        const tr = rowOf(b, w) + d.dr;
        const tc = colOf(b, w) + d.dc;
        const br = rowOf(b, w) - d.dr;
        const bc = colOf(b, w) - d.dc;
        if (tr < 0 || tr >= h || tc < 0 || tc >= w || br < 0 || br >= h || bc < 0 || bc >= w) continue;
        const target = idx(tr, tc, w);
        const behind = idx(br, bc, w);
        if (!floorSet.has(target) || boxSet.has(target)) continue;
        if (!floorSet.has(behind) || boxSet.has(behind)) continue;
        if (!region.set.has(behind)) continue; // player cannot reach the pushing spot
        if (dead.has(target) || forbidden.has(target)) continue;
        const walk = walkPath(node.player, behind, floorSet, boxSet, w, h);
        if (walk === null) continue;
        const nb = [...node.boxes];
        nb[bi] = target;
        nb.sort((a, b2) => a - b2);
        const nbSet = new Set<number>(nb);
        if (frozen(nb)) continue;
        const nRegion = regionOf(b, nbSet);
        const nKey = keyOf(nb, nRegion.min);
        const ng = node.g + 1;
        const prev = nodes.get(nKey);
        if (prev && prev.g <= ng) continue;
        const nh = heuristic(nb);
        nodes.set(nKey, { key: nKey, boxes: nb, player: b, g: ng, h: nh, parent: node.key, move: walk + d.ch });
        heapPush(heap, { f: greedy ? nh : ng + nh, h: nh, seq: seq++, key: nKey });
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Whole-level solving: decompose into rooms, stitch with hallway walks.
// ---------------------------------------------------------------------------

type SolveResult = { moves: string; pushes: number; expansions: number; rooms: number };

function simulate(
  lv: Level,
  moves: string,
): { player: number; boxes: Set<number>; ok: boolean } {
  let player = lv.player;
  const boxes = new Set<number>(lv.boxes);
  for (const ch of moves) {
    const d = DIRS.find((x) => x.ch === ch);
    if (!d) return { player, boxes, ok: false };
    const nr = rowOf(player, lv.w) + d.dr;
    const nc = colOf(player, lv.w) + d.dc;
    if (nr < 0 || nr >= lv.h || nc < 0 || nc >= lv.w) return { player, boxes, ok: false };
    const ni = idx(nr, nc, lv.w);
    if (lv.wall[ni]) return { player, boxes, ok: false };
    if (boxes.has(ni)) {
      const br = nr + d.dr;
      const bc = nc + d.dc;
      if (br < 0 || br >= lv.h || bc < 0 || bc >= lv.w) return { player, boxes, ok: false };
      const bi = idx(br, bc, lv.w);
      if (lv.wall[bi] || boxes.has(bi)) return { player, boxes, ok: false };
      boxes.delete(ni);
      boxes.add(bi);
    }
    player = ni;
  }
  return { player, boxes, ok: true };
}

function countPushes(lv: Level, moves: string): number {
  let player = lv.player;
  const boxes = new Set<number>(lv.boxes);
  let pushes = 0;
  for (const ch of moves) {
    const d = DIRS.find((x) => x.ch === ch) as (typeof DIRS)[number];
    const nr = rowOf(player, lv.w) + d.dr;
    const nc = colOf(player, lv.w) + d.dc;
    const ni = idx(nr, nc, lv.w);
    if (boxes.has(ni)) {
      pushes++;
      const bi = idx(nr + d.dr, nc + d.dc, lv.w);
      boxes.delete(ni);
      boxes.add(bi);
    }
    player = ni;
  }
  return pushes;
}

function solveLevel(lv: Level): SolveResult {
  const empty: SolveResult = { moves: '', pushes: 0, expansions: 0, rooms: 0 };
  if (lv.boxes.every((b) => lv.goals.has(b))) return empty;

  const { rooms, stray } = detectRooms(lv);
  const stats: SearchStats = { expansions: 0, fallback: false };

  // Fallback: whole-board search when the level does not fit the
  // hallway+rooms shape (e.g. hand-made edge levels).
  if (stray.length > 0 || rooms.length === 0) {
    let r = astarSearch(lv.floor, lv.goals, lv.boxes, lv.player, new Set<number>(), WHOLE_ASTAR_CAP, false, lv.w, lv.h, lv.wall, stats);
    if (r === null) {
      stats.fallback = true;
      r = astarSearch(lv.floor, lv.goals, lv.boxes, lv.player, new Set<number>(), WHOLE_GREEDY_CAP, true, lv.w, lv.h, lv.wall, stats);
    }
    if (r === null) return { ...empty, expansions: stats.expansions };
    const sim = simulate(lv, r.moves);
    if (!sim.ok || ![...sim.boxes].every((b) => lv.goals.has(b))) return { ...empty, expansions: stats.expansions };
    return { moves: r.moves, pushes: countPushes(lv, r.moves), expansions: stats.expansions, rooms: 0 };
  }

  // Room-by-room: solve each room from its door, stitch with walks.
  let allMoves = '';
  let curPlayer = lv.player;
  const curBoxes = new Set<number>(lv.boxes);
  const fullFloor = new Set<number>(lv.floor);
  for (const room of rooms) {
    if (room.boxes.length === 0) continue;
    // Walk the player to this room's entry cell (hallway cell outside door).
    const walk = walkPath(curPlayer, room.entry, fullFloor, curBoxes, lv.w, lv.h);
    if (walk === null) return { ...empty, expansions: stats.expansions };
    allMoves += walk;
    curPlayer = room.entry;
    // Sub-board: room cells + the WHOLE hallway row. The player must be
    // able to stand on any hallway cell to push a box near the door (a
    // room's floor can touch the hallway along more than the door cell);
    // every hallway cell is forbidden to boxes, since a box pushed out of
    // its room can never return.
    const hallwayRow = rowOf(lv.player, lv.w);
    const hallwayCells = lv.floor.filter((f) => rowOf(f, lv.w) === hallwayRow);
    const subFloor = [...room.cells, ...hallwayCells].sort((a, b) => a - b);
    const subGoals = new Set<number>(room.goals);
    const forbidden = new Set<number>(hallwayCells);
    let r = astarSearch(subFloor, subGoals, room.boxes, room.entry, forbidden, ROOM_ASTAR_CAP, false, lv.w, lv.h, lv.wall, stats);
    if (r === null) {
      stats.fallback = true;
      r = astarSearch(subFloor, subGoals, room.boxes, room.entry, forbidden, ROOM_GREEDY_CAP, true, lv.w, lv.h, lv.wall, stats);
    }
    if (r === null) return { ...empty, expansions: stats.expansions };
    // Apply the room solution to the full-level state.
    const before = { player: curPlayer, boxes: new Set<number>(curBoxes) };
    const tmpLv: Level = { ...lv, player: before.player, boxes: [...before.boxes].sort((a, b) => a - b) };
    const sim = simulate(tmpLv, r.moves);
    if (!sim.ok) return { ...empty, expansions: stats.expansions };
    allMoves += r.moves;
    curPlayer = sim.player;
    // The room solution was simulated on the full level, so sim.boxes is
    // already the complete, up-to-date box set.
    curBoxes.clear();
    for (const b of sim.boxes) curBoxes.add(b);
  }

  // Verify the stitched solution on the full level before trusting it.
  const finalSim = simulate(lv, allMoves);
  if (!finalSim.ok || ![...finalSim.boxes].every((b) => lv.goals.has(b))) {
    return { ...empty, expansions: stats.expansions };
  }
  return { moves: allMoves, pushes: countPushes(lv, allMoves), expansions: stats.expansions, rooms: rooms.length };
}

const solve = (level: string): SolveResult => solveLevel(parseLevel(level));

for (const id of readdirSync(INPUT_DIR).sort()) {
  const res = solve(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${res.moves}\n`);
  console.error(`${id} pushes=${res.pushes} moves=${res.moves.length} expansions=${res.expansions} rooms=${res.rooms}`);
}
