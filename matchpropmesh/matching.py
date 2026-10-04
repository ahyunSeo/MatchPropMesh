"""One-to-one bipartite matching with a GPU auction solver."""
import torch


@torch.no_grad()
def auction_linear_assignment(cost, max_rounds=12, decay=0.25, patience=100, tol=1e-12):
    """Min-cost assignment for a batch of cost matrices (B,N,M) by epsilon-scaling auction.

    Rows (or columns, if N > M) are matched one-to-one. The inner loop stops early when
    the number of assigned rows stops changing for `patience` iterations.
    Returns lists of matched row and column indices per batch item.
    """
    B, N0, M0 = cost.shape
    dev, dt = cost.device, cost.dtype

    transposed = N0 > M0
    if transposed:
        cost = cost.transpose(1, 2)
    N, M = cost.shape[1], cost.shape[2]

    profit = -cost
    neg_inf = torch.finfo(dt).min
    inner_iters = int(min(20000, max(1000, 3 * N)))
    scale = torch.nanmedian(torch.nanmedian(cost.abs(), dim=2).values, dim=1).values
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    eps = (1e-3 * scale).to(dt)

    prices = torch.zeros(B, M, dtype=dt, device=dev)
    owner_of_item = torch.full((B, M), -1, dtype=torch.long, device=dev)
    item_of_row = torch.full((B, N), -1, dtype=torch.long, device=dev)

    # Greedy nearest-neighbor warm start; prices start at the best-to-second-best gap.
    k2 = 2 if M >= 2 else 1
    best2_vals, best2_idx = torch.topk(profit, k=k2, dim=2)
    best_val, best_j = best2_vals[..., 0], best2_idx[..., 0]
    for b in range(B):
        order = torch.argsort(best_val[b], descending=True)
        seen = torch.full((M,), False, device=dev)
        for i in order.tolist():
            j = best_j[b, i].item()
            if profit[b, i, j] == neg_inf:
                continue
            if not seen[j]:
                seen[j] = True
                owner_of_item[b, j] = i
                item_of_row[b, i] = j
        owned = torch.nonzero(seen, as_tuple=False).squeeze(-1)
        if owned.numel() > 0:
            row_net = profit[b, owner_of_item[b, owned]] - prices[b]
            top2 = torch.topk(row_net, k=k2, dim=1)
            if k2 == 2:
                prices[b, owned] = torch.clamp(top2.values[:, 0] - top2.values[:, 1] + 1e-9, min=0)
            else:
                prices[b, owned] = torch.clamp(top2.values[:, 0] * 0 + 1e-9, min=0)

    for _ in range(max_rounds):
        no_prog = 0
        stop = False
        for _ in range(inner_iters):
            free_rows = item_of_row == -1
            if not free_rows.any():
                break
            prev_assigned = (item_of_row >= 0).sum().item()

            for b in range(B):
                rows = torch.nonzero(free_rows[b], as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                net = profit[b, rows] - prices[b].unsqueeze(0)
                top2 = torch.topk(net, k=k2, dim=1)
                b1, j1 = top2.values[:, 0], top2.indices[:, 0]
                b2 = top2.values[:, 1] if k2 == 2 else torch.full_like(b1, neg_inf)

                good = b1 > (neg_inf / 2)
                if not good.any():
                    continue
                rows, items = rows[good], j1[good]
                offered = prices[b, items] + ((b1[good] - b2[good]) + eps[b])

                # Each item goes to its highest bidder.
                max_offer = torch.full((M,), neg_inf, dtype=dt, device=dev)
                max_offer.scatter_reduce_(0, items, offered, reduce='amax', include_self=True)
                is_winner = offered >= (max_offer[items] - tol)

                # Ties go to the lowest row index (deterministic on GPU).
                first_row_for_item = torch.full((M,), N, dtype=torch.long, device=dev)
                first_row_for_item.scatter_reduce_(0, items[is_winner], rows[is_winner],
                                                   reduce='amin', include_self=True)
                sel_items = torch.nonzero(first_row_for_item < N, as_tuple=False).squeeze(-1)
                if sel_items.numel() == 0:
                    continue
                sel_rows = first_row_for_item[sel_items]

                prices[b, sel_items] = max_offer[sel_items]
                prev = owner_of_item[b, sel_items]
                owner_of_item[b, sel_items] = sel_rows
                item_of_row[b, sel_rows] = sel_items
                freed = prev >= 0
                if freed.any():
                    item_of_row[b, prev[freed]] = -1

            if (item_of_row >= 0).sum().item() == prev_assigned:
                no_prog += 1
                if no_prog >= patience:
                    stop = True
                    break
            else:
                no_prog = 0

        if (item_of_row >= 0).all().item() or stop:
            break
        eps = eps * decay

    rows_out, cols_out = [], []
    for b in range(B):
        rows = torch.nonzero(item_of_row[b] >= 0, as_tuple=False).squeeze(-1)
        cols = item_of_row[b, rows]
        rows_out.append((cols if transposed else rows).clone())
        cols_out.append((rows if transposed else cols).clone())
    return rows_out, cols_out


@torch.no_grad()
def match_points(A, B, cost_thres, batch_rows=1024, batch_cols=4096):
    """One-to-one matching of point sets A (N,3) and B (M,3) under the Euclidean cost.
    Pairs with cost >= cost_thres are discarded after the assignment."""
    N, M = A.shape[0], B.shape[0]
    if N == 0 or M == 0:
        empty = torch.empty(0, dtype=torch.long, device=A.device)
        return empty, empty

    cost = torch.empty((1, N, M), dtype=A.dtype, device=A.device)
    for rs in range(0, N, batch_rows):
        re = min(rs + batch_rows, N)
        for cs in range(0, M, batch_cols):
            ce = min(cs + batch_cols, M)
            diff = A[rs:re].view(re - rs, 1, 3) - B[cs:ce].view(1, ce - cs, 3)
            cost[0, rs:re, cs:ce] = (diff * diff).sum(dim=-1) ** 0.5

    rows, cols = auction_linear_assignment(cost)
    rows, cols = rows[0], cols[0]
    keep = cost[0, rows, cols] < cost_thres
    return rows[keep], cols[keep]
