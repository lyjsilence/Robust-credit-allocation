import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# -------------------- distance helpers --------------------
def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def min_dist_to_set(p, S):
    return min(manhattan(p, q) for q in S) if S else 10**9

def avg_min_dist_to_set(S, T):
    """avg over s in S of min_{t in T} dist(s,t)"""
    ds = []
    for s in S:
        ds.append(min_dist_to_set(s, T))
    return float(np.mean(ds)) if ds else 0.0


# -------------------- grid helpers --------------------
def neighbors(i, j, n, m):
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < n and 0 <= nj < m:
            yield (ni, nj)

def is_connected(cells, n, m):
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        i, j = q.popleft()
        for ni, nj in neighbors(i, j, n, m):
            if (ni, nj) in cells and (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append((ni, nj))
    return len(seen) == len(cells)

def bbox_area(cells):
    rows = [i for i, _ in cells]
    cols = [j for _, j in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)

def largest_component_size(eligible, n, m):
    """eligible: set of cells (i,j)"""
    seen = set()
    best = 0
    for c in eligible:
        if c in seen:
            continue
        q = deque([c])
        seen.add(c)
        comp = 0
        while q:
            i, j = q.popleft()
            comp += 1
            for ni, nj in neighbors(i, j, n, m):
                if (ni, nj) in eligible and (ni, nj) not in seen:
                    seen.add((ni, nj))
                    q.append((ni, nj))
        best = max(best, comp)
    return best


def cells_to_pairs(A, B, cells):
    return [(A[i], B[j]) for (i, j) in sorted(cells)]

# -------------------- test sampler (block-like, connected) --------------------
def sample_test_block(n, m, k=5, seed=None, max_expand=4):
    """
    生成一个“块状/矩形风格”的连通 test，大小为 k。
    做法：先取一个面积>=k、且不太大的矩形块，然后优先从角/边删掉多余格子，
    保持 4-邻接连通，直到剩 k 格。
    """
    rng = random.Random(seed)

    # 1) 选择一个矩形 (h,w)，让面积 area >= k 且 area 不要比 k 大太多
    rects = []
    for h in range(1, n + 1):
        for w in range(1, m + 1):
            area = h * w
            if area >= k and area <= k + max_expand:
                rects.append((h, w, area))
    if not rects:
        # 兜底：允许更大的矩形
        for h in range(1, n + 1):
            for w in range(1, m + 1):
                area = h * w
                if area >= k:
                    rects.append((h, w, area))

    def weight(h, w, area):
        # 更偏向“接近正方形”和“面积更接近 k”的矩形
        compact = 1.0 / (1.0 + abs(h - w))
        area_pen = 1.0 / (1.0 + (area - k))
        return compact * area_pen

    ws = [weight(h, w, area) for (h, w, area) in rects]
    h, w, area = rng.choices(rects, weights=ws, k=1)[0]

    i0 = rng.randrange(0, n - h + 1)
    j0 = rng.randrange(0, m - w + 1)

    cells = {(i, j) for i in range(i0, i0 + h) for j in range(j0, j0 + w)}

    # 2) 从角/边优先删除，直到剩 k 个格子，且保持连通
    while len(cells) > k:
        corners = [(i0, j0), (i0, j0 + w - 1), (i0 + h - 1, j0), (i0 + h - 1, j0 + w - 1)]
        corner_candidates = [c for c in corners if c in cells]

        boundary = []
        for (i, j) in cells:
            deg = sum(((ni, nj) in cells) for (ni, nj) in neighbors(i, j, n, m))
            if deg <= 2:  # 边界/拐角更容易删
                boundary.append((i, j))

        remove_pool = corner_candidates + boundary
        if not remove_pool:
            remove_pool = list(cells)
        rng.shuffle(remove_pool)

        removed = False
        for c in remove_pool:
            new_cells = set(cells)
            new_cells.remove(c)
            if is_connected(new_cells, n, m):
                cells = new_cells
                removed = True
                break

        if not removed:
            # 极端兜底：随便删一个还能连通的
            any_pool = list(cells)
            rng.shuffle(any_pool)
            for c in any_pool:
                new_cells = set(cells)
                new_cells.remove(c)
                if is_connected(new_cells, n, m):
                    cells = new_cells
                    removed = True
                    break
        if not removed:
            break

    # 强制检查
    assert len(cells) == k, f"test size mismatch: expected {k}, got {len(cells)}"
    return cells


# -------------------- connected train under maximal margin --------------------
def compute_dist_to_test(n, m, test_cells):
    dist = {}
    for i in range(n):
        for j in range(m):
            dist[(i, j)] = min_dist_to_set((i, j), test_cells)
    return dist

def sample_connected_cluster_in_eligible(eligible, dist_to_test, n, m, k, rng, max_steps=10000):
    """
    在 eligible 子图上扩展生成一个连通簇（4邻接）大小 k。
    选择策略：优先
      1) 与当前簇相邻边数更多（更块状）
      2) bbox 增量更小（更紧凑）
      3) dist_to_test 更大（更远）
    """
    eligible_list = list(eligible)
    if len(eligible_list) < k:
        return None

    # 起点：从“更远离 test”的点中偏向抽取
    dvals = np.array([dist_to_test[c] for c in eligible_list], dtype=float)
    w = (dvals - dvals.min() + 1.0) ** 2  # 加权偏向远点
    start = rng.choices(eligible_list, weights=w.tolist(), k=1)[0]

    chosen = {start}

    # 维护 bbox
    rmin = rmax = start[0]
    cmin = cmax = start[1]
    cur_area = 1

    steps = 0
    while len(chosen) < k:
        steps += 1
        if steps > max_steps:
            return None

        # frontier：所有已选点的邻居中，仍在 eligible 且未选的
        cand = set()
        for (i, j) in chosen:
            for nb in neighbors(i, j, n, m):
                if nb in eligible and nb not in chosen:
                    cand.add(nb)
        if not cand:
            return None

        scored = []
        for (i, j) in cand:
            # adjacency：与当前簇相邻的边数
            adj = sum((nb in chosen) for nb in neighbors(i, j, n, m))

            # bbox area increase
            nrmin, nrmax = min(rmin, i), max(rmax, i)
            ncmin, ncmax = min(cmin, j), max(cmax, j)
            new_area = (nrmax - nrmin + 1) * (ncmax - ncmin + 1)
            area_inc = new_area - cur_area

            dtest = dist_to_test[(i, j)]
            noise = rng.random() * 1e-3

            # 选择：adj 大优先；area_inc 小优先；dtest 大优先
            scored.append(((adj, -area_inc, dtest, noise), (i, j), (nrmin, nrmax, ncmin, ncmax, new_area)))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_cell = scored[0][1]
        rmin, rmax, cmin, cmax, cur_area = scored[0][2]

        chosen.add(best_cell)

    return chosen


def sample_train_connected_max_margin(n, m, test_cells, k_train=15, seed=None,
                                     n_trials_per_D=3000):
    """
    1) 对每个距离阈值 D（从大到小），构造 eligible={dist_to_test>=D 且不与 test 重叠}
    2) 若 eligible 最大连通分量 < k_train，则该 D 不可行
    3) 在可行 D 下多次随机生成连通簇，选 avgDist 最大、bbox 更紧凑的最优解
    4) 返回最优 train 和 D*
    """
    rng = random.Random(seed)
    dist_to_test = compute_dist_to_test(n, m, test_cells)

    # 候选 D 值（只看非 test 的格子）
    all_cells = [(i, j) for i in range(n) for j in range(m) if (i, j) not in test_cells]
    Ds = sorted({dist_to_test[c] for c in all_cells}, reverse=True)

    best_train = None
    best_D = None
    best_score = None  # (avgDist, -bboxArea)

    for D in Ds:
        eligible = {c for c in all_cells if dist_to_test[c] >= D}

        # 快速不可行剪枝：必须有足够大的连通分量
        if len(eligible) < k_train:
            continue
        if largest_component_size(eligible, n, m) < k_train:
            continue

        # 在该 D 下尝试构造连通 train
        local_best = None
        local_best_score = None

        for _ in range(n_trials_per_D):
            cand = sample_connected_cluster_in_eligible(
                eligible, dist_to_test, n, m, k_train, rng
            )
            if cand is None:
                continue
            # 由于 eligible 已保证 dist>=D 且不包含 test，因此 minDist>=D 自动满足
            avgd = avg_min_dist_to_set(cand, test_cells)
            area = bbox_area(cand)

            score = (avgd, -area)  # 先最大化平均距离，再偏好紧凑形状
            if local_best_score is None or score > local_best_score:
                local_best_score = score
                local_best = cand

        if local_best is not None:
            # 找到可行的最大 D 就可以返回（因为 D 是从大到小遍历）
            best_train = local_best
            best_D = D
            best_score = local_best_score
            break

    if best_train is None:
        raise RuntimeError("No connected train found. Try reducing k_train or increasing trials.")

    return best_train #, best_D, best_score


# -------------------- plotting --------------------
def plot_grid(A, B, test_cells, train_cells, title=None, origin="upper"):
    n, m = len(A), len(B)
    grid = np.zeros((n, m), dtype=int)
    for (i, j) in test_cells:
        grid[i, j] = 1
    for (i, j) in train_cells:
        grid[i, j] = 2

    cmap = ListedColormap(["white", "yellow", "royalblue"])
    plt.figure(figsize=(m * 0.85, n * 0.85))
    plt.imshow(grid, cmap=cmap, aspect="equal", origin=origin, vmin=0, vmax=2)

    plt.xticks(np.arange(m), B)
    plt.yticks(np.arange(n), A)
    plt.xlabel("quarter")
    plt.ylabel("econ")

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        mpatches.Patch(color="yellow", label=f"test ({len(test_cells)})"),
        mpatches.Patch(color="royalblue", label=f"train ({len(train_cells)})"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", frameon=True)

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def cells_to_pairs(A, B, cells):
    return [(A[i], B[j]) for (i, j) in sorted(cells)]


# -------------------- demo --------------------
if __name__ == "__main__":
    A = [1, 2, 3, 4, 5, 6]        # econ
    B = [1, 2, 3, 4, 5, 6, 7]     # quarter
    n, m = len(A), len(B)

    seed = 42

    for seed in range(10):
        # 1) test：块状连通 5 格
        test_cells = sample_test_block(n, m, k=3, seed=seed)

        # 2) train：连通 15 格，且与 test 的最小距离 D* 最大
        train_cells, D_star, (avgd, neg_area) = sample_train_connected_max_margin(
            n, m, test_cells, k_train=16, seed=seed, n_trials_per_D=4000
        )

        # diagnostics
        dist_to_test = compute_dist_to_test(n, m, test_cells)
        min_train_to_test = min(dist_to_test[c] for c in train_cells)
        avg_train_to_test = float(np.mean([dist_to_test[c] for c in train_cells]))

        print("TEST pairs :", cells_to_pairs(A, B, test_cells))
        print("TRAIN pairs:", cells_to_pairs(A, B, train_cells))
        print(f"D* (max feasible minDist with connectivity) = {D_star}")
        print(f"Achieved minDist = {min_train_to_test}, avgDist = {avg_train_to_test:.2f}")
        print("Train connected?", is_connected(train_cells, n, m))

        plot_grid(
            A, B,
            test_cells=test_cells,
            train_cells=train_cells,
            title=f"seed={seed} | D*={D_star} | minDist={min_train_to_test}, avgDist={avg_train_to_test:.2f}",
            origin="upper"  # 想让 econ 从下往上递增用 "lower"
        )