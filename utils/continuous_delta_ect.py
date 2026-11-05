"""
Author: Nello Blaser

As part of Encoding the Euler Characteristic Transform (2026, arXiv) by Blaser et al. 

All credit where credit is due.

-------------------------------------------------------------------------

Continuous Delta ECT — node-attributed Euler Characteristic Transform.

The output has shape [2, N_active, num_dirs]:

    row 0 : filtration threshold (projection of that node)
    row 1 : ECT change at this node (delta = #vertices - #edges + #faces)

Nodes where the ECT change is zero are masked out,
so N_active <= N. Each simplex is attributed to the node that has the
maximum projection in the given direction (ties broken by the lower local
index via ``argmax``).
"""

import torch
from cubical_ect import image_to_cubical_complex


def compute_continuous_delta_ect_edges(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    v: torch.Tensor,
    faces: torch.Tensor | None = None,
    index: torch.Tensor | None = None,
    sort: bool = True,
) -> torch.Tensor | list:
    """Continuous delta ECT for graphs.

    Args:
        x:          Node feature matrix [N, D].
        edge_index: Edge index [2, E] in torch-geometric format.
        v:          Direction matrix [D, num_dirs].
        faces:      Optional face index tensor [3, F] for triangle meshes.
        index:      Optional batch vector [N].
        sort:       If True, sort nodes by ascending threshold per direction.

    Returns:
        [2, N_active, num_dirs] (single graph) or list of [2, N_b_active, num_dirs] (batch).
        Row 0: filtration threshold (projection of the node)
        Row 1: ECT delta (delta = #vertices - #edges + #faces attributed to this node)
        Nodes with delta == 0 in all directions are masked out.
    """
    N = x.shape[0]
    device = x.device
    nh = x @ v  # (N, num_dirs)

    def _responsible_counts(simplex_idx, nh_b, N_b, nd):
        """Scatter simplex attributions: each simplex goes to its max-projection node."""
        k, M = simplex_idx.shape
        h    = nh_b[simplex_idx]                                        # (k, M, nd)
        am   = h.argmax(dim=0)                                          # (M, nd)
        si_T = simplex_idx.T.unsqueeze(-1).expand(M, k, nd)            # (M, k, nd)
        resp = si_T.gather(1, am.unsqueeze(1)).squeeze(1)              # (M, nd)

        d_idx = torch.arange(nd, device=device).unsqueeze(0).expand(M, nd)
        flat  = resp * nd + d_idx
        counts = torch.zeros(N_b * nd, device=device)
        counts.scatter_add_(0, flat.flatten(), torch.ones(M * nd, device=device))
        return counts.view(N_b, nd)

    def _single(nh_b, ei_b, N_b, fi_b):
        nd = nh_b.shape[1]

        ecounts = _responsible_counts(ei_b, nh_b, N_b, nd) if ei_b.shape[1] > 0 \
                  else torch.zeros(N_b, nd, device=device)
        fcounts = _responsible_counts(fi_b, nh_b, N_b, nd) if fi_b is not None and fi_b.shape[1] > 0 \
                  else torch.zeros(N_b, nd, device=device)

        delta = torch.ones(N_b, nd, device=device) - ecounts + fcounts  # (N_b, nd)

        # Mask: keep only nodes with non-zero ECT change in at least one direction
        keep    = (delta != 0).any(dim=1)                               # (N_b,)
        nh_k    = nh_b[keep]
        delta_k = delta[keep]

        result = torch.stack([nh_k, delta_k])                           # (2, N_keep, nd)

        if sort:
            _, sidx = nh_k.sort(dim=0)
            result = result.gather(1, sidx.unsqueeze(0).expand_as(result))
        return result

    if index is None:
        return _single(nh, edge_index, N, faces)

    batch_len = int(index.max()) + 1
    results = []
    for b in range(batch_len):
        node_mask = index == b
        node_ids  = node_mask.nonzero(as_tuple=True)[0]
        remap     = torch.full((N,), -1, dtype=torch.long, device=device)
        remap[node_ids] = torch.arange(node_ids.shape[0], device=device)

        edge_mask = index[edge_index[0]] == b
        sub_ei    = remap[edge_index[:, edge_mask]]

        sub_fi = None
        if faces is not None:
            face_mask = index[faces[0]] == b
            sub_fi    = remap[faces[:, face_mask]]

        nh_b = nh[node_mask]
        results.append(_single(nh_b, sub_ei, nh_b.shape[0], sub_fi))
    return results


def compute_continuous_delta_ect_cubical_batch(
    images: torch.Tensor,
    threshold: float,
    v: torch.Tensor,
    sort: bool = True,
) -> list:
    """Continuous delta ECT for cubical complexes (binary images).

    Args:
        images:    Grayscale images [B, H, W], values in [0, 1].
        threshold: Binarization threshold (pixels >= threshold are foreground).
        v:         Direction matrix [2, num_dirs].
        sort:      If True, sort nodes by ascending threshold per direction.

    Returns:
        List of B tensors, each of shape [2, N_i_active, num_dirs].
        Row 0: filtration threshold (projection of the node)
        Row 1: ECT delta (delta = #vertices - #edges + #faces attributed to this node)
        Nodes with delta == 0 in all directions are masked out.
    """
    num_dirs = v.shape[1]
    results  = []

    for i in range(images.shape[0]):
        mask = images[i] >= threshold
        cx   = image_to_cubical_complex(mask)
        x, edges, squares = cx["x"], cx["edges"], cx["squares"]
        N = x.shape[0]

        if N == 0:
            results.append(torch.zeros(2, 0, num_dirs, device=v.device))
            continue

        nh  = x @ v   # (N, num_dirs)
        dev = nh.device

        def _responsible(simplex_idx):
            k, M  = simplex_idx.shape
            h     = nh[simplex_idx]                                      # (k, M, num_dirs)
            am    = h.argmax(dim=0)                                      # (M, num_dirs)
            idx_T = simplex_idx.T.unsqueeze(-1).expand(M, k, num_dirs)  # (M, k, num_dirs)
            return idx_T.gather(1, am.unsqueeze(1)).squeeze(1)          # (M, num_dirs)

        def _counts(resp, M):
            d_idx = torch.arange(num_dirs, device=dev).unsqueeze(0).expand(M, num_dirs)
            flat  = resp * num_dirs + d_idx
            c     = torch.zeros(N * num_dirs, device=dev)
            c.scatter_add_(0, flat.flatten(), torch.ones(M * num_dirs, device=dev))
            return c.view(N, num_dirs)

        E, S    = edges.shape[1], squares.shape[1]
        ecounts = _counts(_responsible(edges),   E) if E > 0 else torch.zeros(N, num_dirs, device=dev)
        fcounts = _counts(_responsible(squares), S) if S > 0 else torch.zeros(N, num_dirs, device=dev)

        delta = torch.ones(N, num_dirs, device=dev) - ecounts + fcounts  # (N, num_dirs)

        # Mask: keep only nodes with non-zero ECT change in at least one direction
        keep    = (delta != 0).any(dim=1)                                # (N,)
        nh_k    = nh[keep]
        delta_k = delta[keep]

        result = torch.stack([nh_k, delta_k])                            # (2, N_keep, num_dirs)

        if sort:
            _, sidx = nh_k.sort(dim=0)                                   # (N_keep, num_dirs)
            result  = result.gather(1, sidx.unsqueeze(0).expand_as(result))

        results.append(result)

    return results


def discretize_continuous_delta_ect(
    ect: torch.Tensor | list,
    radius: float,
    resolution: int,
    scale: float,
) -> torch.Tensor:
    """Discretizes continuous delta ECT output onto a uniform grid.

    Accepts the output of ``compute_continuous_delta_ect_edges``,
    ``compute_continuous_delta_ect_cubical_batch``, or
    ``compute_continuous_delta_ect_point_cloud``.
    Row 1 of the input is taken directly as the ECT change per node.

    Args:
        ect:        [2, N_active, num_dirs] tensor or list of such tensors.
        radius:     The grid covers [-radius, radius].
        resolution: Number of evenly spaced grid points.
        scale:      Steepness of the sigmoid approximation.

    Returns:
        [resolution, num_dirs] for a single sample, or
        [B, resolution, num_dirs] for a batch.
    """
    def _single(e: torch.Tensor) -> torch.Tensor:
        # e: (2, N_active, num_dirs)
        thresholds = e[0]   # (N_active, num_dirs)
        changes    = e[1]   # (N_active, num_dirs) — pre-computed ECT delta

        lin = torch.linspace(-radius, radius, resolution, device=e.device)
        lin = lin.view(resolution, 1, 1)                              # (res, 1, 1)
        contributions = changes * torch.sigmoid(scale * (lin - thresholds))
        return contributions.sum(dim=1)                               # (res, num_dirs)

    if isinstance(ect, list):
        return torch.stack([_single(e) for e in ect], dim=0)
    return _single(ect)


def compute_continuous_delta_ect_point_cloud(
    x: torch.Tensor,
    v: torch.Tensor,
    sort: bool = True
) -> list:
    """Computes the continuous ECT of a batch of point clouds.

    Args:
        x:
            Point clouds of shape [B, N, D] where B is the batch size,
            N is the number of points and D is the ambient dimension.
        v:
            Direction tensor of shape [D, num_dirs].
        sort: 
            Sort thresholds

    Returns:
        List of B tensors, each of shape [2, N, num_dirs].
            row 0: thresholds where Euler characteristic changes (sorted ascending if sort=True)
            row 1: change in Euler characteristic at corresponding threshold (+1 per point)
    """
    B, N, _ = x.shape
    nh = x @ v  # (B, N, num_dirs)
    signs = torch.ones(N, device=x.device)

    if not sort: 
        results = []
        for b in range(B):
            h_b = nh[b]  
            s_b = signs.unsqueeze(-1).expand_as(h_b)
            results.append(torch.stack((h_b, s_b)))
        return results

    results = []
    for b in range(B):
        h_b = nh[b]  # (N, num_dirs)
        sorted_h, sort_idx = h_b.sort(dim=0)
        sorted_s = signs[sort_idx]
        results.append(torch.stack((sorted_h, sorted_s)))
    return results

