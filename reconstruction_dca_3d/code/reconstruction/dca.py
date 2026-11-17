"""Joint Difference-of-Convex Algorithm (DCA) with Cauchy IRLS & local SO(3) Gauss-Newton.

Implements the Joint DCA optimization framework (Section 11 & Appendix B of the
reference paper):
  1. Cauchy robust loss on point observations -> IRLS weights w_i = 1/(1+(e_i/c)^2).
  2. Local SO(3) tangent polarization around current camera orientations g_v -> exp([w]_x) g_v.
  3. Matrix-free damped Gauss-Newton subproblem (Eq. 34):
       (J^T W J + mu * D) d = -g,  D = diag(J^T W J) + D_reg
     solved via preconditioned conjugate gradient (PCG) without dense Hessians.
  4. Adaptive trust-region damping: reject/infeasible -> mu *= 4; accept -> mu /= 3
     under F(theta + d) <= F(theta) - 1e-4 * mu * ||d||_D^2.
  5. Graduated Non-Convexity (Homotopy) decreasing Cauchy scale c = 30 -> 10 -> 3 px.
  6. Preserves IPC CCD safe_fraction and valid() self-intersection / camera-plane guards.
"""
import copy
import time
import numpy as np
from .engine import queries, point_residuals, first_hit_constraints
from .losses import sample_field, visibility_term
from .point_terms import cauchy_irls_weights


def _pack(d, active_X=None):
    dx = d['X'] if active_X is None else d['X'][active_X]
    return np.concatenate([
        dx.ravel(),
        d['C'].ravel(),
        d['g'][1:].ravel(),
        d['lf'].ravel(),
        d['pp'].ravel(),
        d['h'].ravel(),
    ])


def _unpack(vec, P, K, active_X=None):
    idx = 0
    if active_X is None:
        nx = P * 3
        dx = vec[idx:idx + nx].reshape(P, 3)
        idx += nx
    else:
        n_act = int(np.sum(active_X))
        nx = n_act * 3
        dx = np.zeros((P, 3), float)
        dx[active_X] = vec[idx:idx + nx].reshape(n_act, 3)
        idx += nx
    nc = K * 3
    dC = vec[idx:idx + nc].reshape(K, 3)
    idx += nc
    ng = max(K - 1, 0) * 3
    dg = np.zeros((K, 3), float)
    if K > 1:
        dg[1:] = vec[idx:idx + ng].reshape(K - 1, 3)
    idx += ng
    dlf = vec[idx:idx + K].copy()
    idx += K
    dpp = vec[idx:idx + K * 2].reshape(K, 2)
    idx += K * 2
    dh = vec[idx:idx + 3].copy()
    return dict(X=dx, C=dC, g=dg, lf=dlf, pp=dpp, h=dh)


class _GaussNewtonContext:
    """Matrix-free Gauss-Newton Hessian-vector product & diagonal preconditioner D_k."""

    def __init__(self, s, a, grad_current):
        self.s = s
        self.a = a
        self.X = s['X']
        self.C = s['C']
        self.g = s['g']
        self.f = np.exp(s['lf'])
        self.pp = s['pp']
        self.K, self.P = len(self.C), len(self.X)
        K, P = self.K, self.P
        width = a['width']
        faces = a['faces']

        self.q = np.einsum('kij,kpj->kpi', self.g, self.X[None] - self.C[:, None])
        self.z_q = self.q[..., 2]
        self.uv_q = self.f[:, None, None] * self.q[..., :2] / self.z_q[..., None]
        projected = self.uv_q + self.pp[:, None]

        # Diagonal accumulators in tangent coordinates
        diag_X = np.zeros((P, 3), float)
        diag_C = np.zeros((K, 3), float)
        diag_g = np.zeros((K, 3), float)
        diag_lf = np.zeros(K, float)
        diag_pp = np.zeros((K, 2), float)

        # --- 1. Point reprojection Gauss-Newton ---
        ci, pi = a['ci'], a['pi']
        self.has_points = len(ci) > 0
        if self.has_points:
            self.ci, self.pi = ci, pi
            xt = queries(s, a)
            self.qt = np.einsum('kij,ktj->kti', self.g, xt[None] - self.C[:, None])
            self.z_t = self.qt[ci, pi, 2]
            self.u_rel = self.f[ci, None] * self.qt[ci, pi, :2] / self.z_t[:, None]
            res = self.u_rel + self.pp[ci] - a['uv']
            tol_px = a.get('point_tolerance_pixels', 0.)
            if a.get('point_weights') is not None:
                w_pt = np.asarray(a['point_weights'], float)
            elif a.get('cauchy_scale_pixels') is not None:
                w_pt = cauchy_irls_weights(res, width, a['cauchy_scale_pixels'], tol_px)
            else:
                w_pt = np.ones(len(ci), float)
            base_pt = 3.0 / len(ci)  # Hessian of 1.5 * mean(w * ||r||^2)
            self.pt_tol_px = tol_px
            if tol_px > 0.:
                dist = np.linalg.norm(res, axis=1)
                tol = tol_px / width
                act = dist > tol
                safe_d = np.maximum(dist, 1e-15)
                self.pt_n = res / safe_d[:, None]
                self.pt_w_rad = base_pt * w_pt * act
                self.pt_w_tan = base_pt * w_pt * act * np.maximum(dist - tol, 0.) / safe_d
                # Effective 2x2 weight diagonal on (u, v)
                n2 = self.pt_n**2
                w_diag2 = self.pt_w_rad[:, None] * n2 + self.pt_w_tan[:, None] * (1.0 - n2)
            else:
                self.pt_w_iso = base_pt * w_pt
                w_diag2 = np.repeat(self.pt_w_iso[:, None], 2, axis=1)

            # Accumulate point diagonal into tangent variables
            np.add.at(diag_pp, ci, w_diag2)
            np.add.at(diag_lf, ci, np.sum(w_diag2 * (self.u_rel**2), axis=1))
            # A = f/z * [I_2, -u_rel/f] = [f/z, 0, -u0/z; 0, f/z, -u1/z]
            inv_z = 1.0 / self.z_t
            fz = self.f[ci] * inv_z
            A = np.zeros((len(ci), 2, 3), float)
            A[:, 0, 0] = fz
            A[:, 1, 1] = fz
            A[:, :, 2] = -self.u_rel * inv_z[:, None]
            # wrt X via g[ci] and barycentrics
            Ag = np.einsum('irc,icj->irj', A, self.g[ci])
            d_xt = np.einsum('ir,irj->ij', w_diag2, Ag * Ag)
            np.add.at(diag_C, ci, d_xt)
            # wrt omega: dqt = w x qt -> row j of [qt]_x^T A^T
            qx_norm2 = np.zeros((len(ci), 3), float)
            for j in range(3):
                e = np.zeros(3); e[j] = 1.0
                c_vec = np.cross(e, self.qt[ci, pi])  # (N, 3)
                Ac = np.einsum('irc,ic->ir', A, c_vec)
                qx_norm2[:, j] = np.sum(w_diag2 * Ac * Ac, axis=1)
            np.add.at(diag_g, ci, qx_norm2)
            ab = a['anchor_bary']
            av = a['anchor_vertices']
            for corner in range(3):
                np.add.at(diag_X, av[pi, corner], d_xt * (ab[pi, corner:corner + 1]**2))

        # --- 2. Silhouette Out & In Gauss-Newton ---
        T = len(faces)
        bary = a['bary']
        S = len(bary)
        samples = np.einsum('sj,ktjd->ktsd', bary, projected[:, faces]).reshape(K, T * S, 2)
        dist_out, fg = sample_field(a['fields'], samples, width, a.get('field_coefficients'))
        outside = np.maximum(dist_out - a['pixel_tolerance'], 0.)
        c_sil = a.get('silhouette_cauchy_pixels')
        if c_sil is not None and c_sil > 0:
            cn = float(c_sil) / width
            w_sil_out = 1.0 / (1.0 + (outside / cn)**2)
        else:
            w_sil_out = 1.0
        self.fg = fg  # (K, T*S, 2)
        self.w_out_s = (2.0 / (K * T * S)) * (dist_out > a['pixel_tolerance']) * w_sil_out

        # Reverse silhouette (targets inside mask)
        p_t = a['targets']
        M = p_t.shape[1]
        tri = projected[:, faces]
        A_t, B_t, C_t = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        v0 = B_t - A_t; v1 = C_t - A_t; v2 = p_t[:, :, None, :] - A_t[:, None]
        den = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
        safe = np.where(np.abs(den) > 1e-14, den, 1.)
        u_b = (v2[..., 0] * v1[:, None, :, 1] - v2[..., 1] * v1[:, None, :, 0]) / safe[:, None]
        v_b = (v0[:, None, :, 0] * v2[..., 1] - v0[:, None, :, 1] * v2[..., 0]) / safe[:, None]
        inside = ((u_b >= 0) & (v_b >= 0) & (u_b + v_b <= 1) & (np.abs(den[:, None]) > 1e-14)).any(axis=-1)
        starts = tri; ends = np.roll(tri, -1, axis=2); edge = ends - starts
        rel = p_t[:, :, None, None, :] - starts[:, None]
        t_proj = np.clip(np.sum(rel * edge[:, None], axis=-1) / np.maximum(np.sum(edge**2, axis=-1)[:, None], 1e-15), 0, 1)
        closest = starts[:, None] + t_proj[..., None] * edge[:, None]
        delta = closest - p_t[:, :, None, None, :]
        d2 = np.sum(delta**2, axis=-1)
        which = d2.reshape(K, M, -1).argmin(axis=-1)
        fi = which // 3; ei = which % 3
        kk = np.arange(K)[:, None]; mm = np.arange(M)[None]
        d_in = np.sqrt(d2[kk, mm, fi, ei])
        in_act = (~inside) & (d_in > a['pixel_tolerance'])
        vk_act, vm_act = np.nonzero(in_act)
        self.has_in = len(vk_act) > 0
        if self.has_in:
            fi_a = fi[vk_act, vm_act]; ei_a = ei[vk_act, vm_act]
            self.in_vk = vk_act
            self.in_va = faces[fi_a, ei_a]
            self.in_vb = faces[fi_a, (ei_a + 1) % 3]
            self.in_t = t_proj[vk_act, vm_act, fi_a, ei_a]
            self.in_n = delta[vk_act, vm_act, fi_a, ei_a] / np.maximum(d_in[vk_act, vm_act, None], 1e-15)
            if c_sil is not None and c_sil > 0:
                act_d = np.maximum(d_in[vk_act, vm_act] - a['pixel_tolerance'], 0.)
                w_sil_in = 1.0 / (1.0 + (act_d / cn)**2)
            else:
                w_sil_in = np.ones(len(vk_act), float)
            self.in_w = (2.0 / (K * M)) * w_sil_in

        # Accumulate silhouette diagonal wrt projected[K, P, 2]
        diag_proj = np.zeros((K, P, 2), float)
        w_fg2 = (self.w_out_s[..., None] * (fg * fg)).reshape(K, T, S, 2)
        for corner in range(3):
            b2 = bary[:, corner][None, None, :, None]**2
            contrib = np.sum(w_fg2 * b2, axis=2)  # (K, T, 2)
            for v in range(K):
                np.add.at(diag_proj[v], faces[:, corner], contrib[v])
        if self.has_in:
            wn2 = self.in_w[:, None] * (self.in_n**2)
            np.add.at(diag_proj, (self.in_vk, self.in_va), wn2 * ((1.0 - self.in_t)[:, None]**2))
            np.add.at(diag_proj, (self.in_vk, self.in_vb), wn2 * (self.in_t[:, None]**2))

        # Chain diag_proj[K, P, 2] -> tangent coordinates
        diag_pp += diag_proj.sum(axis=1)
        diag_lf += np.sum(diag_proj * (self.uv_q**2), axis=(1, 2))
        inv_zq = 1.0 / self.z_q
        fz_q = self.f[:, None] * inv_zq  # (K, P)
        # J_proj wrt q_k,p: [[fz_q, 0, -uv0/z], [0, fz_q, -uv1/z]]
        dq_diag = np.zeros((K, P, 3), float)
        dq_diag[..., 0] = diag_proj[..., 0] * (fz_q**2)
        dq_diag[..., 1] = diag_proj[..., 1] * (fz_q**2)
        dq_diag[..., 2] = np.sum(diag_proj * (self.uv_q**2), axis=-1) * (inv_zq**2)
        # X & C diagonal via g[k]: row j of g^T diag(dq_diag) g -> sum_c dq_diag[c] * g[k, c, j]^2
        d_X_sil = np.einsum('kpc,kcj->kpj', dq_diag, self.g * self.g)
        diag_X += d_X_sil.sum(axis=0)
        diag_C += d_X_sil.sum(axis=1)
        # omega_k diagonal: sum_{p, r} diag_proj[k, p, r] * (J_proj[r] * (e_j x q[k,p]))^2
        for j in range(3):
            e = np.zeros(3); e[j] = 1.0
            c_vec = np.cross(e, self.q)  # (K, P, 3)
            dp0 = fz_q * c_vec[..., 0] - (self.uv_q[..., 0] * inv_zq) * c_vec[..., 2]
            dp1 = fz_q * c_vec[..., 1] - (self.uv_q[..., 1] * inv_zq) * c_vec[..., 2]
            diag_g[:, j] += np.sum(diag_proj[..., 0] * (dp0**2) + diag_proj[..., 1] * (dp1**2), axis=1)

        # --- 3. First-hit visibility Gauss-Newton (when active constraints/duals exist) ---
        self.has_vis = False
        if self.has_points:
            va = dict(
                ci=ci, pi=P + pi, faces=faces, relative_tolerance=.001, base_weight=2.,
                return_constraints=True, penalty=a.get('visibility_penalty', 0.),
                dual=a.get('visibility_dual', np.zeros(len(ci))),
                feasibility_tolerance=a.get('visibility_feasibility_tolerance', 0.),
            )
            q_all = np.concatenate([self.q, self.qt], axis=1)
            _, _, _, constraints = visibility_term(q_all, va)
            rho = va.get('penalty', 0.)
            dual = np.asarray(va.get('dual', np.zeros(len(ci))), float)
            act_vis = constraints > 0
            if rho > 0:
                act_vis = act_vis | ((dual + rho * constraints) > 0)
            if np.any(act_vis):
                self._setup_visibility_gn(q_all, va, constraints, diag_X, diag_C, diag_g, a)

        # --- 4. Smoothness, Intrinsic Priors & IPC Contact Barrier Hessian ---
        Y = self.X - self.X.mean(0)
        self.scale_X = float(np.mean(np.sum(Y**2, axis=1))) + 1e-12
        self.L = a['L']
        self.smooth_coef = 2.0 * a['smooth_weight'] / (P * self.scale_X)
        L_col_sq = np.asarray(np.sum(self.L * self.L, axis=0)).ravel() if hasattr(self.L, 'toarray') else np.sum(self.L**2, axis=0)
        diag_X += self.smooth_coef * L_col_sq[:, None]

        cw = a.get('calibration_weight', 1. / width**2)
        pw = cw / a.get('principal_sigma', .25)**2
        fw = cw / a.get('log_focal_sigma', np.log(2.))**2
        self.pp_coef = 2.0 * pw / s['pp'].size
        self.lf_coef = 2.0 * fw / K
        diag_pp += self.pp_coef
        diag_lf += self.lf_coef

        self.has_contact = False
        cm = a.get('contact_model')
        if cm is not None and hasattr(cm, 'hessian_data'):
            H_ipc, Y_ipc, r_ipc = cm.hessian_data(self.X)
            if H_ipc.nnz > 0:
                self.has_contact = True
                self.contact_weight = float(a.get('contact_weight', 1. / width**2))
                self.H_ipc = H_ipc
                self.Y_ipc = Y_ipc
                self.r_ipc = float(r_ipc)
                diag_X += self.contact_weight * H_ipc.diagonal().reshape(P, 3) / (self.r_ipc * self.r_ipc)

        # Regularization & trust-region curvature floor so occluded/nonlinear-prior vertices never teleport
        pos_DX = diag_X[diag_X > 0]
        med_DX = float(np.median(pos_DX)) if len(pos_DX) else (1.0 / (width * width))
        prior_floor = max(2.0 * a.get('prior_weight', 1e-3) / (P * self.scale_X), 1e-8 / (width * width))
        gx_norm = np.linalg.norm(grad_current['X'], axis=1)
        tr_floor = np.maximum(prior_floor, np.maximum(0.02 * med_DX, gx_norm / 0.12))[:, None]
        diag_X = np.maximum(diag_X, tr_floor)
        cam_floor = max(1e-6 / (width * width), float(np.median(diag_C)) * 1e-4 if np.any(diag_C > 0) else 1e-6)
        diag_C = np.maximum(diag_C, cam_floor)
        diag_g = np.maximum(diag_g, max(1e-6 / (width * width), float(np.median(diag_g[1:])) * 1e-4 if K > 1 and np.any(diag_g[1:] > 0) else 1e-6))
        diag_g[0] = 0.
        diag_lf = np.maximum(diag_lf, 1e-6 / (width * width))
        diag_pp = np.maximum(diag_pp, 1e-6 / (width * width))
        diag_h = np.full(3, max(2.0 * a.get('prior_weight', 1e-3), 1e-6 / (width * width)))

        self.diag_dict = dict(X=diag_X, C=diag_C, g=diag_g, lf=diag_lf, pp=diag_pp, h=diag_h)
        self.D_vec = None  # built lazily with active_X

    def _setup_visibility_gn(self, q_all, va, constraints, diag_X, diag_C, diag_g, a):
        ci, pi, faces = va['ci'], va['pi'], va['faces']
        tri = q_all[ci][:, faces]
        D = q_all[ci, pi]
        e1 = tri[:, :, 1] - tri[:, :, 0]
        e2 = tri[:, :, 2] - tri[:, :, 0]
        h = np.cross(D[:, None], e2)
        det = np.sum(e1 * h, axis=-1)
        valid = np.abs(det) > 1e-12
        inv = np.zeros_like(det)
        inv[valid] = 1. / det[valid]
        S = -tri[:, :, 0]
        u = inv * np.sum(S * h, axis=-1)
        cross = np.cross(S, e1)
        v = inv * np.sum(D[:, None] * cross, axis=-1)
        t = inv * np.sum(e2 * cross, axis=-1)
        incident = (faces[None] == pi[:, None, None]).any(axis=-1)
        rho = va.get('penalty', 0.)
        dual = np.asarray(va.get('dual', np.zeros(len(ci))), float)
        ok = valid & ~incident & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (t > 1e-7)
        if not rho:
            ok &= (t < 1)
        candidates = np.where(ok, t, np.inf)
        face = candidates.argmin(axis=1)
        rows = np.arange(len(ci))
        near = candidates[rows, face]
        act_mask = constraints > 0
        if rho > 0:
            act_mask = act_mask | ((dual + rho * constraints) > 0)
        act_mask = act_mask & np.isfinite(near)
        active = np.flatnonzero(act_mask)
        if not len(active):
            return
        self.has_vis = True
        f = face[active]
        self.vis_views = ci[active]
        self.vis_q_idx = pi[active] - self.P  # index into qt
        self.vis_faces = faces[f]
        normal = np.cross(e1[active, f], e2[active, f])
        denom = np.sum(normal * D[active], axis=-1)
        self.vis_normal = normal / denom[:, None]
        self.vis_near = near[active]
        self.vis_alpha = np.column_stack([1. - u[active, f] - v[active, f], u[active, f], v[active, f]])
        w_base = np.where(constraints[active] > 0, 2.0 * va.get('base_weight', 1.), 0.0)
        if rho > 0:
            w_al = np.where((dual[active] + rho * constraints[active]) > 0, rho, 0.0)
        else:
            w_al = np.zeros(len(active), float)
        self.vis_w = (w_base + w_al) / len(ci)

        # Accumulate visibility curvature into D_k so PCG & mu-damping remain well-scaled
        ab = a['anchor_bary']
        av = a['anchor_vertices']
        n_world = np.einsum('nij,nj->ni', self.g[self.vis_views], self.vis_normal)
        dq_D2 = self.vis_w[:, None] * ((self.vis_near[:, None] * n_world)**2)
        np.add.at(diag_C, self.vis_views, dq_D2)
        for corner in range(3):
            np.add.at(diag_X, av[self.vis_q_idx, corner], dq_D2 * (ab[self.vis_q_idx, corner:corner + 1]**2))
            fac_c = self.vis_w[:, None] * ((self.vis_alpha[:, corner:corner + 1] * n_world)**2)
            np.add.at(diag_X, self.vis_faces[:, corner], fac_c)
            np.add.at(diag_C, self.vis_views, fac_c)
        # Angular sensitivity: n_cam . (e_j x (near*D - sum alpha_c face_c))
        for j in range(3):
            e = np.zeros(3); e[j] = 1.0
            q_D = self.qt[self.vis_views, self.vis_q_idx]
            d_c_D = -self.vis_near * np.sum(self.vis_normal * np.cross(e, q_D), axis=-1)
            q_tri = np.sum(self.vis_alpha[..., None] * self.q[self.vis_views[:, None], self.vis_faces], axis=1)
            d_c_tri = np.sum(self.vis_normal * np.cross(e, q_tri), axis=-1)
            np.add.at(diag_g[:, j], self.vis_views, self.vis_w * ((d_c_D + d_c_tri)**2))

    def apply_H(self, d):
        """Apply symmetric PSD Gauss-Newton operator H_GN to tangent step dict d."""
        K, P = self.K, self.P
        dX, dC, dg, dlf, dpp = d['X'], d['C'], d['g'], d['lf'], d['pp']
        dg_full = dg.copy()
        dg_full[0] = 0.

        # dq = g @ (dX - dC) + dg x q
        dq = np.einsum('kij,kpj->kpi', self.g, dX[None] - dC[:, None]) + np.cross(dg_full[:, None], self.q)
        inv_zq = 1.0 / self.z_q
        d_proj = (
            (self.f[:, None] * inv_zq)[..., None] * dq[..., :2]
            - (self.uv_q * inv_zq[..., None]) * dq[..., 2:3]
            + self.uv_q * dlf[:, None, None]
            + dpp[:, None]
        )

        # Adjoint accumulators
        g_proj = np.zeros((K, P, 2), float)
        gX = np.zeros((P, 3), float)
        gC = np.zeros((K, 3), float)
        gg = np.zeros((K, 3), float)
        glf = np.zeros(K, float)
        gpp = np.zeros((K, 2), float)

        # 1. Silhouette Out
        faces = self.a['faces']
        bary = self.a['bary']
        d_samp = np.einsum('sj,ktjd->ktsd', bary, d_proj[:, faces]).reshape(K, len(faces) * len(bary), 2)
        d_dist = np.sum(self.fg * d_samp, axis=-1)
        g_samp = (self.w_out_s * d_dist)[..., None] * self.fg
        g_tri = np.einsum('ktsd,sj->ktjd', g_samp.reshape(K, len(faces), len(bary), 2), bary)
        for corner in range(3):
            for v in range(K):
                np.add.at(g_proj[v], faces[:, corner], g_tri[v, :, corner])

        # 2. Silhouette In
        if self.has_in:
            d_closest = (1.0 - self.in_t)[:, None] * d_proj[self.in_vk, self.in_va] + self.in_t[:, None] * d_proj[self.in_vk, self.in_vb]
            d_rad = np.sum(self.in_n * d_closest, axis=-1)
            g_closest = (self.in_w * d_rad)[:, None] * self.in_n
            np.add.at(g_proj, (self.in_vk, self.in_va), g_closest * (1.0 - self.in_t)[:, None])
            np.add.at(g_proj, (self.in_vk, self.in_vb), g_closest * self.in_t[:, None])

        # Map g_proj -> gq, glf, gpp
        glf += np.sum(g_proj * self.uv_q, axis=(1, 2))
        gpp += g_proj.sum(axis=1)
        gq = np.zeros_like(self.q)
        gq[..., :2] = g_proj * (self.f[:, None] * inv_zq)[..., None]
        gq[..., 2] = -np.sum(g_proj * self.uv_q, axis=-1) * inv_zq

        # 3. Point reprojection & Visibility
        if self.has_points:
            ci, pi = self.ci, self.pi
            ab = self.a['anchor_bary']
            av = self.a['anchor_vertices']
            d_xt = np.einsum('tj,tjd->td', ab, dX[av])
            d_qt = np.einsum('kij,ktj->kti', self.g, d_xt[None] - dC[:, None]) + np.cross(dg_full[:, None], self.qt)
            d_qobs = d_qt[ci, pi]
            inv_zt = 1.0 / self.z_t
            d_pred = (
                (self.f[ci] * inv_zt)[:, None] * d_qobs[:, :2]
                - (self.u_rel * inv_zt[:, None]) * d_qobs[:, 2:3]
                + self.u_rel * dlf[ci, None]
                + dpp[ci]
            )
            if self.pt_tol_px > 0.:
                rad_comp = np.sum(self.pt_n * d_pred, axis=1, keepdims=True) * self.pt_n
                tan_comp = d_pred - rad_comp
                g_u = self.pt_w_rad[:, None] * rad_comp + self.pt_w_tan[:, None] * tan_comp
            else:
                g_u = self.pt_w_iso[:, None] * d_pred

            np.add.at(glf, ci, np.sum(g_u * self.u_rel, axis=1))
            np.add.at(gpp, ci, g_u)
            g_dq = np.zeros_like(d_qobs)
            g_dq[:, :2] = g_u * (self.f[ci] * inv_zt)[:, None]
            g_dq[:, 2] = -np.sum(g_u * self.u_rel, axis=1) * inv_zt
            d_qt_adj = np.zeros_like(self.qt)
            np.add.at(d_qt_adj, (ci, pi), g_dq)

            if self.has_vis:
                # d_constraint = -vis_near * (vis_n . d_D) + vis_n . sum(alpha_c * dq_face_c)
                d_D = d_qt[self.vis_views, self.vis_q_idx]
                d_tri = np.sum(self.vis_alpha[..., None] * dq[self.vis_views[:, None], self.vis_faces], axis=1)
                d_c = -self.vis_near * np.sum(self.vis_normal * d_D, axis=-1) + np.sum(self.vis_normal * d_tri, axis=-1)
                factor = self.vis_w * d_c
                np.add.at(d_qt_adj, (self.vis_views, self.vis_q_idx), -factor[:, None] * self.vis_near[:, None] * self.vis_normal)
                for corner in range(3):
                    np.add.at(gq, (self.vis_views, self.vis_faces[:, corner]), factor[:, None] * self.vis_alpha[:, corner, None] * self.vis_normal)

            gC -= np.einsum('kij,ki->kj', self.g, d_qt_adj.sum(1))
            gg += np.cross(self.qt, d_qt_adj).sum(1)
            g_xt = np.einsum('kij,kti->ktj', self.g, d_qt_adj).sum(0)
            for corner in range(3):
                np.add.at(gX, av[:, corner], g_xt * ab[:, corner:corner + 1])

        gX += np.einsum('kij,kpi->kpj', self.g, gq).sum(0)
        gC -= np.einsum('kij,ki->kj', self.g, gq.sum(1))
        gg += np.cross(self.q, gq).sum(1)
        gg[0] = 0.

        # 4. Smoothness, Intrinsic priors & IPC Contact Barrier
        gX += self.smooth_coef * (self.L.T @ (self.L @ dX))
        if self.has_contact:
            dXc = dX - dX.mean(0)
            dY = (dXc - self.Y_ipc * (np.sum(self.Y_ipc * dXc) / P)) / self.r_ipc
            HY = np.asarray(self.H_ipc.dot(dY.ravel())).reshape(P, 3)
            gX += self.contact_weight * (HY - HY.mean(0) - self.Y_ipc * (np.sum(self.Y_ipc * HY) / P)) / self.r_ipc
        gpp += self.pp_coef * dpp
        glf += self.lf_coef * dlf
        return dict(X=gX, C=gC, g=gg, lf=glf, pp=gpp, h=np.zeros(3, float))

    def solve_subproblem(self, grad_vec, D_vec, mu, active_X=None, only_X=False, max_cg=25, cg_tol=1e-3):
        """Solve (H_GN + mu * D) d = -grad via Jacobi-preconditioned CG."""
        P, K = self.P, self.K
        if only_X:
            nx = (int(np.sum(active_X)) if active_X is not None else P) * 3
            if nx == 0:
                return np.zeros_like(grad_vec)
            b_x = -grad_vec[:nx]
            Dx = D_vec[:nx]
            inv_Mx = 1.0 / np.maximum((1.0 + mu) * Dx, 1e-18)

            def matvec_x(vx):
                v_full = np.zeros_like(grad_vec)
                v_full[:nx] = vx
                d_dict = _unpack(v_full, P, K, active_X)
                hv = _pack(self.apply_H(d_dict), active_X)[:nx]
                return hv + mu * (Dx * vx)

            x = np.zeros(nx, float)
            r = b_x.copy()
            z = inv_Mx * r
            p = z.copy()
            rz = float(np.dot(r, z))
            b_norm = float(np.linalg.norm(b_x))
            if b_norm > 0. and np.isfinite(rz):
                for _ in range(max_cg):
                    Ap = matvec_x(p)
                    pAp = float(np.dot(p, Ap))
                    if pAp <= 0. or not np.isfinite(pAp):
                        break
                    alpha = rz / pAp
                    x = x + alpha * p
                    r = r - alpha * Ap
                    if np.linalg.norm(r) <= cg_tol * b_norm:
                        break
                    z = inv_Mx * r
                    rz_new = float(np.dot(r, z))
                    if rz_new <= 0.:
                        break
                    p = z + (rz_new / rz) * p
                    rz = rz_new
            out = np.zeros_like(grad_vec)
            out[:nx] = x
            return out

        b = -grad_vec
        d_scale = np.maximum((1.0 + mu) * D_vec, 1e-18)
        # h block has zero H_GN curvature, so exact diagonal is mu * D_h
        d_scale[-3:] = np.maximum(mu * D_vec[-3:], 1e-18)
        inv_M = 1.0 / d_scale

        def matvec(v):
            d_dict = _unpack(v, P, K, active_X)
            hv = _pack(self.apply_H(d_dict), active_X)
            return hv + mu * (D_vec * v)

        x = np.zeros_like(b)
        r = b.copy()
        z = inv_M * r
        p = z.copy()
        rz = float(np.dot(r, z))
        b_norm = float(np.linalg.norm(b))
        if b_norm == 0. or not np.isfinite(rz):
            return x
        for _ in range(max_cg):
            Ap = matvec(p)
            pAp = float(np.dot(p, Ap))
            if pAp <= 0. or not np.isfinite(pAp):
                break
            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            if np.linalg.norm(r) <= cg_tol * b_norm:
                break
            z = inv_M * r
            rz_new = float(np.dot(r, z))
            if rz_new <= 0.:
                break
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x


def _grazing_visibility_dofs(s, trial, a, newly):
    """Identify anchor + occluding face vertices and cameras for newly violated rays."""
    ci = a['ci'][newly]
    pi_q = a['pi'][newly]
    verts = set(np.unique(a['anchor_vertices'][pi_q]).tolist())
    q = np.einsum('kij,kpj->kpi', trial['g'], trial['X'][None] - trial['C'][:, None])
    xt = queries(trial, a)
    qt = np.einsum('kij,ktj->kti', trial['g'], xt[None] - trial['C'][:, None])
    q_all = np.concatenate([q, qt], axis=1)
    pi = len(trial['X']) + pi_q
    faces = a['faces']
    tri = q_all[ci][:, faces]
    D = q_all[ci, pi]
    e1 = tri[:, :, 1] - tri[:, :, 0]
    e2 = tri[:, :, 2] - tri[:, :, 0]
    h = np.cross(D[:, None], e2)
    det = np.sum(e1 * h, axis=-1)
    ok_det = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok_det] = 1. / det[ok_det]
    S = -tri[:, :, 0]
    u = inv * np.sum(S * h, axis=-1)
    cross = np.cross(S, e1)
    v = inv * np.sum(D[:, None] * cross, axis=-1)
    t = inv * np.sum(e2 * cross, axis=-1)
    inc = (faces[None] == pi[:, None, None]).any(axis=-1)
    ok = ok_det & ~inc & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (t > 1e-7)
    if not a.get('visibility_penalty', 0.):
        ok &= (t < 1)
    cand = np.where(ok, t, np.inf)
    f_hit = cand.argmin(axis=1)
    hit_ok = np.isfinite(cand[np.arange(len(ci)), f_hit])
    if np.any(hit_ok):
        verts.update(np.unique(faces[f_hit[hit_ok]]).tolist())
    return np.array(sorted(verts), int), np.unique(ci)


def _run_dca_stage(s, a, gauge, steps, verbose, objective_fn, step_fn, valid_fn,
                   mu_init=0.5, mu_min=1e-5, mu_max=1e7, max_cg=25):
    if steps < 1:
        raise ValueError('Positive step count required for DCA.')
    active_X = getattr(objective_fn, 'free_mask', None)
    only_X = getattr(objective_fn, 'only_X', False)
    P, K = len(s['X']), len(s['C'])

    def eval_safe(state):
        try:
            return objective_fn(state, a, gauge)
        except (ValueError, FloatingPointError):
            return None

    current = eval_safe(s)
    if current is None:
        raise ValueError('Initial DCA state invalid.')

    mu = float(mu_init)
    objective_scale = a.get('optimization_scale', 1.)
    history = []
    losses = [float(current[0])]
    start = time.perf_counter()
    status = 'max_steps'
    accepted = 0
    rejected = 0
    evaluations = 1
    ccd_limited = 0
    minimum_ccd_fraction = 1.
    mu_history = [mu]
    iteration = 0
    grazing_boost = np.ones((P, 1), float)
    view_boost = np.ones((K, 1), float)

    if only_X and active_X is not None and not np.any(active_X):
        history.append(dict(step=0, **current[2]))
        return dict(
            state=s, stats=current[2], history=history, steps=0, status='small_gradient',
            seconds=time.perf_counter() - start, a=a, gauge=gauge,
            optimization=dict(
                method='Joint DCA (Cauchy IRLS + local SO(3) Gauss-Newton + adaptive mu + CCD)',
                accepted_steps=0, rejected_trials=0, ccd_limited_trials=0,
                minimum_ccd_fraction=1.0, objective_evaluations=1,
                accepted_losses=losses, final_mu=mu, monotone=True,
            ),
        )

    for iteration in range(1, steps + 1):
        loss, gradient, stats = current
        if iteration == 1 or iteration % 25 == 0:
            history.append(dict(step=iteration, **stats))
        if verbose and (iteration == 1 or iteration % 25 == 0):
            cw_str = f', w_med {stats["cauchy_weight_median"]:.3f}' if 'cauchy_weight_median' in stats else ''
            print(f' DCA {iteration}: points {stats["point_rmse_px"]:.3f}px, out {stats["model_to_image_px"]:.3f}px, '
                  f'in {stats["image_to_model_px"]:.3f}px, blocked {stats["visibility_violations"]}, mu {mu:.3g}{cw_str}',
                  flush=True)

        grad_vec = _pack(gradient, active_X)
        if only_X:
            nx = (int(np.sum(active_X)) if active_X is not None else P) * 3
            grad_vec[nx:] = 0.
        g_inf = float(np.max(np.abs(grad_vec))) if len(grad_vec) else 0.0
        if g_inf == 0. or (iteration > 15 and g_inf < 3e-8 / objective_scale):
            status = 'small_gradient'
            break

        gn = _GaussNewtonContext(s, a, gradient)
        gn.diag_dict['X'] *= grazing_boost
        gn.diag_dict['C'] *= view_boost
        gn.diag_dict['g'] *= view_boost
        D_vec = _pack(gn.diag_dict, active_X)
        if only_X:
            nx = (int(np.sum(active_X)) if active_X is not None else P) * 3
            D_vec[nx:] = 1e12

        step_accepted = False
        grazing_skips = 0
        for _ in range(24):
            d_vec = gn.solve_subproblem(grad_vec, D_vec, mu, active_X=active_X, only_X=only_X, max_cg=max_cg)
            d_dict = _unpack(d_vec, P, K, active_X)
            # Clip boundary pushes on lf and pp, then re-pack d_vec to keep ||d||_D exact
            d_dict['lf'] = np.where((s['lf'] <= np.log(.4) + 1e-12) & (d_dict['lf'] < 0), 0., d_dict['lf'])
            d_dict['lf'] = np.where((s['lf'] >= np.log(12.) - 1e-12) & (d_dict['lf'] > 0), 0., d_dict['lf'])
            d_dict['pp'] = np.where((s['pp'] <= -.8 + 1e-12) & (d_dict['pp'] < 0), 0., d_dict['pp'])
            d_dict['pp'] = np.where((s['pp'] >= .8 - 1e-12) & (d_dict['pp'] > 0), 0., d_dict['pp'])
            d_vec = _pack(d_dict, active_X)
            if only_X:
                nx = (int(np.sum(active_X)) if active_X is not None else P) * 3
                d_vec[nx:] = 0.

            fraction = 1.
            if a.get('contact_model') is not None:
                fraction = a['contact_model'].safe_fraction(s['X'], s['X'] + d_dict['X'])
                if fraction < 1.:
                    ccd_limited += 1
                    minimum_ccd_fraction = min(minimum_ccd_fraction, fraction)
                if fraction <= 0.:
                    mu = min(mu * 4.0, mu_max)
                    rejected += 1
                    continue
                d_dict = {key: fraction * val for key, val in d_dict.items()}
                d_vec = d_vec * fraction

            trial = step_fn(s, d_dict, a, gauge)
            if valid_fn is not None and not valid_fn(trial, a):
                mu = min(mu * 4.0, mu_max)
                rejected += 1
                continue

            candidate = eval_safe(trial)
            evaluations += 1
            if candidate is None or not np.isfinite(candidate[0]):
                mu = min(mu * 4.0, mu_max)
                rejected += 1
                continue

            d_D_norm2 = float(np.dot(d_vec, D_vec * d_vec))
            # Teacher's sufficient decrease rule: F_new <= F - 1e-4 * mu * ||d||_D^2
            if candidate[0] <= loss - 1e-4 * mu * d_D_norm2:
                s = trial
                current = candidate
                losses.append(float(candidate[0]))
                accepted += 1
                mu = max(mu / 3.0, mu_min)
                grazing_boost = np.maximum(1.0, grazing_boost * 0.7)
                view_boost = np.maximum(1.0, view_boost * 0.7)
                mu_history.append(mu)
                step_accepted = True
                break
            else:
                # If a grazing visibility discontinuity jumped 0->1 violation, damp
                # the newly-violating anchor/hit-face vertices (and views) persistently
                # so global mu stays small and remaining DOFs keep converging smoothly.
                if candidate[2].get('visibility_violations', 0) > stats.get('visibility_violations', 0) and len(a.get('ci', [])):
                    c_cur = first_hit_constraints(s, a)
                    c_tri = first_hit_constraints(trial, a)
                    newly = np.flatnonzero((c_tri > 0) & (c_cur <= 0))
                    if len(newly):
                        bad_verts, bad_views = _grazing_visibility_dofs(s, trial, a, newly)
                        grazing_boost[bad_verts] = np.minimum(grazing_boost[bad_verts] * 1e4, 1e12)
                        gn.diag_dict['X'][bad_verts] *= 1e4
                        if grazing_skips >= 1 or mu >= 10.0:
                            view_boost[bad_views] = np.minimum(view_boost[bad_views] * 1e2, 1e8)
                            gn.diag_dict['C'][bad_views] *= 1e2
                            gn.diag_dict['g'][bad_views] *= 1e2
                        D_vec = _pack(gn.diag_dict, active_X)
                        if only_X:
                            nx = (int(np.sum(active_X)) if active_X is not None else P) * 3
                            D_vec[nx:] = 1e12
                        if grazing_skips < 4:
                            grazing_skips += 1
                            rejected += 1
                            continue
                mu = min(mu * 4.0, mu_max)
                rejected += 1

        if not step_accepted:
            status = 'mu_max_reached' if mu >= mu_max * 0.99 else 'no_descent_step'
            break

        if len(losses) > 25 and losses[-25] - losses[-1] < 1e-7 * max(abs(losses[-25]), 1e-12):
            status = 'stalled_cost'
            break

    history.append(dict(step=iteration, **current[2]))
    return dict(
        state=s,
        stats=current[2],
        history=history,
        steps=iteration,
        status=status,
        seconds=time.perf_counter() - start,
        a=a,
        gauge=gauge,
        optimization=dict(
            method='Joint DCA (Cauchy IRLS + local SO(3) Gauss-Newton + adaptive mu + CCD)',
            accepted_steps=accepted,
            rejected_trials=rejected,
            ccd_limited_trials=ccd_limited,
            minimum_ccd_fraction=minimum_ccd_fraction,
            objective_evaluations=evaluations,
            accepted_losses=losses,
            final_mu=mu,
            monotone=bool(np.all(np.diff(losses) <= 0)),
        ),
    )


def optimize_dca(s, a, gauge=None, steps=180, verbose=True, objective_fn=None, step_fn=None,
                 valid_fn=None, learning_rates=None, cauchy_schedule=None, mu_init=0.5, max_cg=25):
    """Joint DCA solver supporting fixed-c or graduated non-convexity Homotopy (c = 30 -> 10 -> 3 px)."""
    if objective_fn is None or step_fn is None:
        raise ValueError('Objective and step hooks required.')
    if cauchy_schedule is None:
        if a.get('cauchy_scale_pixels') is not None:
            cauchy_schedule = (float(a['cauchy_scale_pixels']),)
        elif a.get('cauchy_schedule') is not None:
            cauchy_schedule = tuple(float(c) for c in a['cauchy_schedule'])
        else:
            cauchy_schedule = (30., 10., 3.)

    # Filter positive scales and allocate outer step budget across homotopy stages
    scales = [float(c) for c in cauchy_schedule if c > 0]
    if not scales:
        raise ValueError('At least one positive Cauchy scale is required.')
    n_stages = len(scales)
    if n_stages == 1:
        work_a = dict(a, cauchy_scale_pixels=scales[0])
        return _run_dca_stage(
            s, work_a, gauge, steps, verbose, objective_fn, step_fn, valid_fn,
            mu_init=mu_init, max_cg=max_cg,
        )

    # Homotopy schedule: split outer steps smoothly (e.g. 25% / 35% / 40% for 3 stages)
    weights = np.linspace(1.0, 1.6, n_stages)
    weights /= weights.sum()
    alloc = [max(1, int(round(steps * w))) for w in weights]
    alloc[-1] = max(1, steps - sum(alloc[:-1]))

    work_s = copy.deepcopy(s)
    total_start = time.perf_counter()
    all_history = []
    stage_records = []
    total_steps = 0
    total_accepted = 0
    total_rejected = 0
    total_evals = 0
    total_ccd = 0
    min_ccd = 1.0
    last_res = None

    cur_mu = float(mu_init)
    for idx, (c_px, stage_steps) in enumerate(zip(scales, alloc)):
        if verbose:
            print(f'DCA Homotopy stage {idx + 1}/{n_stages}: cauchy_scale={c_px:g}px ({stage_steps} steps)', flush=True)
        work_a = dict(a, cauchy_scale_pixels=c_px)
        res = _run_dca_stage(
            work_s, work_a, gauge, stage_steps, verbose, objective_fn, step_fn, valid_fn,
            mu_init=cur_mu, max_cg=max_cg,
        )
        cur_mu = max(1e-3, min(res['optimization']['final_mu'], 10.0))
        work_s = res['state']
        for h in res['history']:
            all_history.append(dict(homotopy_c_px=c_px, step_total=total_steps + h['step'], **h))
        opt = res['optimization']
        stage_records.append(dict(
            cauchy_scale_pixels=c_px,
            steps=res['steps'],
            status=res['status'],
            accepted_steps=opt['accepted_steps'],
            rejected_trials=opt['rejected_trials'],
            final_mu=opt['final_mu'],
            monotone=opt['monotone'],
            final_loss=opt['accepted_losses'][-1],
        ))
        total_steps += res['steps']
        total_accepted += opt['accepted_steps']
        total_rejected += opt['rejected_trials']
        total_evals += opt['objective_evaluations']
        total_ccd += opt['ccd_limited_trials']
        min_ccd = min(min_ccd, opt['minimum_ccd_fraction'])
        last_res = res

    return dict(
        state=work_s,
        stats=last_res['stats'],
        history=all_history,
        steps=total_steps,
        status=last_res['status'],
        seconds=time.perf_counter() - total_start,
        a=last_res['a'],
        gauge=gauge,
        optimization=dict(
            method='Joint DCA with Homotopy (Cauchy IRLS + local SO(3) Gauss-Newton + adaptive mu + CCD)',
            homotopy_scales_px=scales,
            homotopy_stages=stage_records,
            accepted_steps=total_accepted,
            rejected_trials=total_rejected,
            ccd_limited_trials=total_ccd,
            minimum_ccd_fraction=min_ccd,
            objective_evaluations=total_evals,
            accepted_losses=last_res['optimization']['accepted_losses'],
            final_mu=last_res['optimization']['final_mu'],
            monotone=all(st['monotone'] for st in stage_records),
        ),
    )
