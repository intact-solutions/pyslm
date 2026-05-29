import numpy as np
from scipy.spatial import cKDTree

def update_pv_two_stage(
    xyz_old, p_old, v_old,
    xyz_new, p_new, v_new,
    match_tol=1e-6,
    neighbor_radius=1e-3,
    k_min=3,
    eps=1e-12
):
    tree = cKDTree(xyz_old)

    M = xyz_new.shape[0]

    updated_p = np.copy(p_new)
    updated_v = np.copy(v_new)

    matched = np.zeros(M, dtype=bool)
    matched_idx = np.full(M, -1, dtype=int)

    unmatched_points = []

    # -------------------------
    # Phase 1: Matching
    # -------------------------
    dists_nn, idxs_nn = tree.query(xyz_new, k=1)

    for i in range(M):
        if dists_nn[i] <= match_tol:
            matched[i] = True
            matched_idx[i] = idxs_nn[i]
        else:
            unmatched_points.append((i, xyz_new[i], dists_nn[i]))

    # Report unmatched
    if unmatched_points:
        print(f"[INFO] {len(unmatched_points)} points have no match within tolerance:")
        for i, pt, d in unmatched_points:
            print(f"  idx={i}, xyz={pt}, nearest_dist={d:.3e}")

    # -------------------------
    # Phase 2: Replacement
    # -------------------------
    matched_ids = np.where(matched)[0]
    updated_p[matched_ids] = p_old[matched_idx[matched_ids]]
    updated_v[matched_ids] = v_old[matched_idx[matched_ids]]

    # -------------------------
    # Phase 3: Interpolation / Fallback
    # -------------------------
    for i in np.where(~matched)[0]:
        pt = xyz_new[i]

        # radius neighbors
        neighbors = tree.query_ball_point(pt, r=neighbor_radius)

        if len(neighbors) > 0:
            nbr_xyz = xyz_old[neighbors]
            nbr_p = p_old[neighbors]
            nbr_v = v_old[neighbors]

            # z constraint
            z_center = pt[2]
            mask = nbr_xyz[:, 2] <= z_center

            if np.any(mask):
                nbr_xyz = nbr_xyz[mask]
                nbr_p = nbr_p[mask]
                nbr_v = nbr_v[mask]

                dists = np.linalg.norm(nbr_xyz - pt, axis=1)
                dists = np.maximum(dists, eps)

                weights = 1.0 / dists
                weights /= weights.sum()

                updated_p[i] = np.dot(weights, nbr_p)
                updated_v[i] = np.dot(weights, nbr_v)
                continue  # success → skip fallback

        # -------------------------
        # Fallback: nearest neighbor
        # -------------------------
        nn_idx = idxs_nn[i]
        updated_p[i] = p_old[nn_idx]
        updated_v[i] = v_old[nn_idx]

        print(f"[FALLBACK] idx={i}, xyz={pt} → using nearest neighbor")

    return updated_p, updated_v