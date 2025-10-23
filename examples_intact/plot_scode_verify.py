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


def _square_corners_from_midline(x1, y1, x2, y2):
    # Compute oriented square corners given midpoints of opposite edges
    dx, dy = (x2 - x1), (y2 - y1)
    side = (dx * dx + dy * dy) ** 0.5
    if side == 0:
        # Degenerate; return a tiny box
        eps = 1e-6
        return [(x1 - eps, y1 - eps), (x1 + eps, y1 - eps), (x1 + eps, y1 + eps), (x1 - eps, y1 + eps)]
    ux, uy = dx / side, dy / side  # unit along midline
    vx, vy = -uy, ux               # perpendicular unit
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    h = side * 0.5
    # Four corners: C +/- (u*h) +/- (v*h)
    corners = [
        (cx - ux * h - vx * h, cy - uy * h - vy * h),
        (cx + ux * h - vx * h, cy + uy * h - vy * h),
        (cx + ux * h + vx * h, cy + uy * h + vy * h),
        (cx - ux * h + vx * h, cy - uy * h + vy * h),
    ]
    return corners


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

    # Draw all islands as oriented squares (light gray fill), and entry->exit midline
    for r in island_rows:
        idx = r['idx']
        corners = _square_corners_from_midline(r['x1'], r['y1'], r['x2'], r['y2'])
        xs = [p[0] for p in corners] + [corners[0][0]]
        ys = [p[1] for p in corners] + [corners[0][1]]
        ax.fill(xs, ys, color='#dddddd', alpha=0.4, zorder=0)
        ax.plot(xs, ys, color='#aaaaaa', linewidth=0.8, alpha=0.9, zorder=1)
        # Entry->exit midline
        ax.plot([r['x1'], r['x2']], [r['y1'], r['y2']], color='#999999', linewidth=0.7, alpha=0.7, zorder=2)
        # Annotation: idx and total_time at center
        cx, cy = (r['x1'] + r['x2']) * 0.5, (r['y1'] + r['y2']) * 0.5
        ax.text(cx, cy, f"{idx}\n{r['total_time']:.3f}s", color='#555555', fontsize=4.5, ha='center', va='center')

    # Draw the path segments color-coded by island-idx
    for s in path_segments:
        color = idx_to_color.get(s['idx'], '#d62728')
        ax.plot([s['x1'], s['x2']], [s['y1'], s['y2']], color=color, linewidth=0.9, alpha=0.9, zorder=3)


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
