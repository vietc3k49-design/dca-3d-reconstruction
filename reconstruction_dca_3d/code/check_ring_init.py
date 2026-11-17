"""Verification checks for Phase 2: 360-degree ring feature matching, pose-graph chaining, and multi-view track gradients."""
import copy
import numpy as np
from reconstruction.inputs import feature_pairs
from reconstruction.initialization import initialize, camera_hypotheses, build_uniform_ring_state
from reconstruction.engine import arrays, objective, valid
from reconstruction.mesh import sphere
from reconstruction.raster import render_mesh
from reconstruction.rotations import exp_rotation, look_at


def _synthetic_ring_fixture(num_views=8, size=256):
    """Render a rigid asymmetric textured sphere-topology mesh around a 360-degree ring."""
    X_true, faces = sphere(2)
    X_true = X_true.copy()
    X_true[:, 0] *= 0.85
    X_true[:, 1] *= 1.15
    X_true[:, 2] *= 0.70
    # Deterministic smooth bumpy landmarks so SIFT keypoints lock onto true 3D features
    rng = np.random.default_rng(7)
    X_true += 0.03 * np.sin(5.0 * X_true[:, [1, 2, 0]])
    # Synthetic textured image via per-face projection + deterministic pattern
    centers = X_true[faces].mean(axis=1)
    u_up = np.array([0., -1., 0.])
    C_true = np.zeros((num_views, 3))
    g_true = np.zeros((num_views, 3, 3))
    f_px = 2.0 * size
    pp_px = np.array([size / 2.0, size / 2.0])
    images = []
    masks = []
    for v in range(num_views):
        theta = 2.0 * np.pi * v / num_views
        R = exp_rotation(theta * u_up)
        C_true[v] = R @ np.array([0., -0.35, -3.2])
        # Build camera basis consistent with g[0]=I rotated by R around u_up
        g_true[v] = look_at(C_true[v], np.zeros(3))
        # Ensure image-up is positive -y_world for consistency
        if g_true[v, 1, 1] < 0:
            g_true[v, 0] *= -1.0
            g_true[v, 1] *= -1.0
        im = render_mesh(X_true, faces, C_true[v], g_true[v], f_px, pp_px, (size, size))
        arr = np.array(im)
        mask = np.any(arr != [235, 239, 241], axis=2)
        images.append(arr)
        masks.append(mask)
    return X_true, faces, C_true, g_true, np.array(images), np.array(masks)


def _synthetic_ring_pairs(X_true, C_true, g_true, size, f_norm=2.0, num_views=8):
    """Generate clean pairwise 2D-2D correspondences along circular neighbors."""
    # Sample ~24 well-spaced 3D surface landmarks
    rng = np.random.default_rng(42)
    pts = X_true[rng.choice(len(X_true), 32, replace=False)]
    pairs = []
    # Check visibility by dot product with camera optical center ray
    for va in range(num_views):
        vb = (va + 1) % num_views
        vis_a = []
        vis_b = []
        for P in pts:
            qa = g_true[va] @ (P - C_true[va])
            qb = g_true[vb] @ (P - C_true[vb])
            # Visible from front hemisphere relative to origin
            if (P @ (-C_true[va]) > 0.15) and (P @ (-C_true[vb]) > 0.15) and qa[2] > 0.2 and qb[2] > 0.2:
                pa = (f_norm * qa[:2] / qa[2] + 0.5) * size
                pb = (f_norm * qb[:2] / qb[2] + 0.5) * size
                vis_a.append(pa)
                vis_b.append(pb)
        if len(vis_a) >= 8:
            pairs.append(dict(
                view_a=min(va, vb),
                view_b=max(va, vb),
                pixels_a=np.array(vis_a if va < vb else vis_b),
                pixels_b=np.array(vis_b if va < vb else vis_a),
                inliers=len(vis_a),
            ))
    return pairs


def check_ring_candidate_pairs():
    dummy_imgs = np.zeros((6, 64, 64, 3), np.uint8)
    dummy_masks = np.ones((6, 64, 64), bool)
    pairs_star, info_star = feature_pairs(dummy_imgs, dummy_masks, pair_mode='star')
    assert pairs_star == [] and info_star['pair_mode'] == 'star' and info_star['candidate_pairs'] == 5
    pairs_ring, info_ring = feature_pairs(dummy_imgs, dummy_masks, pair_mode='ring', ring_window=2)
    assert info_ring['pair_mode'] == 'ring' and info_ring['candidate_pairs'] == 12
    print('PASS circular candidate pair generation and empty-keypoint guard')


def check_pose_graph_and_tracks():
    K = 8
    size = 256
    X_true, faces, C_true, g_true, images, masks = _synthetic_ring_fixture(num_views=K, size=size)
    pairs = _synthetic_ring_pairs(X_true, C_true, g_true, size, num_views=K)
    assert len(pairs) == K, f'Expected {K} ring pairs, got {len(pairs)}'

    state, init_faces, init_info = initialize(images, masks, pairs, init_mode='ring', ring_direction='auto')
    assert init_info['uncalibrated_views'] == [], init_info['uncalibrated_views']
    assert len(init_info['calibrated_views']) == K
    assert np.allclose(state['g'] @ state['g'].transpose(0, 2, 1), np.eye(3), atol=1e-8)

    # Check camera centers span full 360 degrees around the object center
    centre = state['X'].mean(0)
    u_up = -state['g'][0, 1]
    r0 = state['C'][0] - centre
    r0h = r0 - (r0 @ u_up) * u_up
    eb = r0h / np.linalg.norm(r0h)
    er = np.cross(u_up, eb)
    angles = []
    for v in range(K):
        rv = state['C'][v] - centre
        rvh = rv - (rv @ u_up) * u_up
        angles.append(float(np.degrees(np.arctan2(rvh @ er, rvh @ eb))) % 360.0)
    diffs = [((angles[(v + 1) % K] - angles[v] + 180.0) % 360.0) - 180.0 for v in range(K)]
    mean_step = abs(np.mean(diffs))
    assert 35.0 < mean_step < 55.0, f'Expected ~45 deg steps around ring, got diffs={diffs}'

    # Build arrays and verify Union-Find multi-view tracks & non-zero v_ref ray_anchors
    state['h'] = np.eye(3)
    a = arrays(images, masks, pairs, state, init_faces)
    assert valid(state, a)
    assert len(set(zip(a['ci'].tolist(), a['pi'].tolist()))) == len(a['ci'])
    # Tracks not observed in view 0 MUST still exist (back-side tracks!)
    tracks_with_v0 = set(a['pi'][a['ci'] == 0].tolist())
    all_tracks = set(a['pi'].tolist())
    back_tracks = all_tracks - tracks_with_v0
    assert len(back_tracks) > 0, 'Expected multi-view tracks strictly on back views (not seen by view 0)'

    # Finite-difference gradient check on multi-view ring objective (including back-side tracks)
    _, grad, _ = objective(state, a)
    worst = 0.0
    eps = 1e-7
    for key in ('X', 'C', 'g', 'lf', 'pp', 'h'):
        values = grad[key]
        indices = [i for i in np.ndindex(values.shape) if not (key == 'g' and i[0] == 0)]
        sampled = indices[::max(1, len(indices) // 12)]
        for idx in sampled:
            sp = copy.deepcopy(state)
            sm = copy.deepcopy(state)
            if key in ('g', 'h'):
                dp = np.zeros((K, 3) if key == 'g' else (3,))
                dp[idx] = eps
                sp[key] = exp_rotation(dp) @ state[key]
                sm[key] = exp_rotation(-dp) @ state[key]
            else:
                sp[key][idx] += eps
                sm[key][idx] -= eps
            err = abs((objective(sp, a)[0] - objective(sm, a)[0]) / (2 * eps) - values[idx])
            worst = max(worst, err)
    assert worst < 2e-6, f'FD gradient mismatch on ring tracks: {worst}'

    # Check partial-SIFT fallback + uniform_ring hypotheses
    partial_pairs = [p for p in pairs if p['view_a'] == 0 and p['view_b'] == 1]
    p_state, p_faces, p_info = initialize(images, masks, partial_pairs, init_mode='ring')
    p_state['h'] = np.eye(3)
    assert len(p_info['uncalibrated_views']) == K - 2
    starts = camera_hypotheses(
        p_state, images=images, masks=masks, init_mode='ring',
        ring_direction='auto', ring_direction_detected=bool(p_info['ring_direction_detected']),
    )
    labels = [lbl for lbl, _ in starts]
    assert any(lbl.startswith('uniform_ring_') for lbl in labels), labels
    for lbl, seed in starts:
        pa = arrays(images, masks, partial_pairs, seed, p_faces)
        assert valid(seed, pa), f'Hypothesis {lbl} invalid'

    ring_only, r_faces = build_uniform_ring_state(images, masks, direction=1.0)
    ring_only['h'] = np.eye(3)
    assert valid(ring_only, arrays(images, masks, [], ring_only, r_faces))
    print(f'PASS 360-ring pose graph ({K} views, step={mean_step:.1f} deg), {len(back_tracks)} back-only tracks, FD err {worst:.3g}, and uniform ring fill-in')


if __name__ == '__main__':
    check_ring_candidate_pairs()
    check_pose_graph_and_tracks()
