import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def read_paths_scode(path: Path):
    segs = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            x1, y1, x2, y2, z, power, speed, idx = parts[:8]
            try:
                segs.append({
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'z': float(z), 'power': float(power), 'speed': float(speed), 'idx': int(float(idx))
                })
            except Exception:
                pass
    return segs


def read_islands_scode(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            x1, y1, x2, y2, z, power, eq_speed, total_time, idx = parts[:9]
            try:
                rows.append({
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2), 'z': float(z),
                    'power': float(power), 'eq_speed': float(eq_speed), 'total_time': float(total_time),
                    'idx': int(float(idx))
                })
            except Exception:
                pass
    return rows


def _convex_hull(points):
    # Monotonic chain convex hull; points: list of (x, y)
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def plot_from_scode(ax, island_rows, path_segments):
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#f0f0f0', linewidth=0.5)

    # Color islands consistently by index
    if island_rows:
        all_idxs = sorted({r['idx'] for r in island_rows})
    else:
        all_idxs = sorted({s['idx'] for s in path_segments})
    cmap = mpl.colormaps.get_cmap('coolwarm')
    idx_to_color = {idx: cmap(0.5 if len(all_idxs) <= 1 else (i / (max(1, len(all_idxs) - 1)))) for i, idx in enumerate(all_idxs)}

    # Draw entry->exit lines for all islands from Query 2 file
    for r in island_rows:
        color = idx_to_color.get(r['idx'], (0.6, 0.6, 0.6, 1.0))
        ax.plot([r['x1'], r['x2']], [r['y1'], r['y2']], color=color, linewidth=0.8, alpha=0.6)
        # Label island index near midpoint
        mx, my = (r['x1'] + r['x2']) * 0.5, (r['y1'] + r['y2']) * 0.5
        ax.text(mx, my, str(r['idx']), color='#666666', fontsize=4, ha='center', va='center')

    # Outline convex hulls for islands present in the paths file
    seg_by_idx = {}
    for s in path_segments:
        seg_by_idx.setdefault(s['idx'], []).append(s)

    for idx, segs in seg_by_idx.items():
        pts = []
        for s in segs:
            pts.append((s['x1'], s['y1']))
            pts.append((s['x2'], s['y2']))
        hull = _convex_hull(pts)
        if len(hull) >= 3:
            xs = [p[0] for p in hull] + [hull[0][0]]
            ys = [p[1] for p in hull] + [hull[0][1]]
            ax.fill(xs, ys, color='#ffcc80', alpha=0.25, zorder=0)
            ax.plot(xs, ys, color='#ffcc80', linewidth=1.0, alpha=0.8)

    # Draw the path segments color-coded by island-idx
    for s in path_segments:
        color = idx_to_color.get(s['idx'], '#d62728')
        ax.plot([s['x1'], s['x2']], [s['y1'], s['y2']], color=color, linewidth=0.9, alpha=0.9)


def main():
    parser = argparse.ArgumentParser(description='Plot existing .scode files: neighborhood paths over island info (no regeneration)')
    parser.add_argument('--paths', type=str, required=True, help='Path to neighborhood paths .scode (Query 1)')
    parser.add_argument('--islands', type=str, required=True, help='Path to island info .scode (Query 2)')
    args = parser.parse_args()

    paths_file = Path(args.paths)
    islands_file = Path(args.islands)
    if not paths_file.exists():
        print('Paths .scode not found:', paths_file)
        return
    if not islands_file.exists():
        print('Islands .scode not found:', islands_file)
        return

    path_segments = read_paths_scode(paths_file)
    island_rows = read_islands_scode(islands_file)

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_from_scode(ax, island_rows, path_segments)
    ax.set_title(f'Verify paths: {paths_file.name}\nIslands: {islands_file.name}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
