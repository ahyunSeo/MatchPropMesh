"""Smoothness propagation of sparse handle displacements over the mesh."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def cotangent_laplacian(V, F, cot_clip=50.0, eps=1e-20):
    """Cotangent Laplacian L = D - W with cotangents clipped to [0, cot_clip], so that
    degenerate or obtuse triangles contribute zero or bounded weight."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = V.shape[0]
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    def cot_at(a, b, c):
        e1, e2 = b - a, c - a
        nrm = np.linalg.norm(np.cross(e1, e2), axis=1)
        dot = np.einsum('ij,ij->i', e1, e2)
        cot = np.zeros_like(dot)
        mask = nrm > eps
        cot[mask] = dot[mask] / nrm[mask]
        return np.clip(cot, 0.0, cot_clip)

    w_jk = 0.5 * cot_at(V[i], V[j], V[k])
    w_ki = 0.5 * cot_at(V[j], V[k], V[i])
    w_ij = 0.5 * cot_at(V[k], V[i], V[j])

    rows = np.concatenate([i, j, j, k, k, i])
    cols = np.concatenate([j, i, k, j, i, k])
    data = np.concatenate([w_ij, w_ij, w_jk, w_jk, w_ki, w_ki])
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W


def boundary_vertices(F):
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return np.empty(0, dtype=np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    E.sort(axis=1)
    UE, counts = np.unique(E, axis=0, return_counts=True)
    return np.unique(UE[counts == 1].ravel()).astype(np.int64)


def propagate_displacements(V, F, handle_idx, handle_disp, tau_rel=1e-6, gamma_rel=1e-3):
    """Solves for the free-vertex displacements u_f that minimize the Dirichlet energy
    with the handles fixed:  (L_ff + tau I + gamma S^T S) u_f = -L_fh d_h,
    where S selects free boundary vertices. tau and gamma are relative to mean(diag(L_ff)).

    V: (N,3) float32 tensor, F: (M,3) faces, handle_idx: (H,), handle_disp: (H,3).
    Returns the deformed vertices V + u as a float32 tensor.
    """
    V = V.numpy()
    handle_idx = np.asarray(handle_idx, dtype=int)
    handle_disp = handle_disp.numpy()
    N = V.shape[0]

    is_handle = np.zeros(N, dtype=bool)
    is_handle[handle_idx] = True
    free_idx = np.where(~is_handle)[0]
    if free_idx.size == 0:
        out = V.copy()
        out[handle_idx] += handle_disp
        return torch.from_numpy(out)

    L = cotangent_laplacian(V, F).tocsr()
    A = L[free_idx[:, None], free_idx].tocsr()
    A_fh = L[free_idx[:, None], handle_idx].tocsr()

    mean_diag = float(A.diagonal().mean())
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        mean_diag = 1.0
    tau = tau_rel * mean_diag
    if tau > 0:
        A = A + sp.diags(np.full(A.shape[0], tau))

    # Weakly anchor free boundary vertices of open meshes.
    bnd = np.setdiff1d(boundary_vertices(F), handle_idx, assume_unique=False)
    if bnd.size:
        S = sp.coo_matrix((np.ones(bnd.size), (np.arange(bnd.size), bnd)), shape=(bnd.size, N)).tocsr()
        S_f = S[:, free_idx]
        A = A + gamma_rel * mean_diag * (S_f.T @ S_f)

    rhs = np.column_stack([-(A_fh @ handle_disp[:, c]) for c in range(3)])
    u_f = spla.factorized(A.copy().tocsc())(rhs)

    u = np.zeros_like(V)
    u[handle_idx] = handle_disp
    u[free_idx] = u_f
    return torch.from_numpy(V + u)
