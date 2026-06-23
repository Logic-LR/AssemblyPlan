"""Diagnostic: understand why Tree F1 is stuck at 0.396/0.034."""
import sys, torch, json
from pathlib import Path
from assembly_plan.data import load_dataset, split_dataset
from assembly_plan.model import build_model
from assembly_plan.decoder import greedy_decode, group_greedy_decode, MergeState
from assembly_plan.evaluate import build_tree_from_list, eval_tree, nonleaf_nodes

def main():
    data_json = Path("experiments/svg_assembly/datasets/tree_generation_dataset.json")
    svg_dir = Path("experiments/svg_assembly/simplified_svg")
    device = torch.device("cuda")

    objects = load_dataset(data_json, svg_dir)
    train_records, test_records = split_dataset(objects)
    part_feat_dim = len(train_records[0].part_features[0].feature_vec)

    model = build_model(part_feat_dim=part_feat_dim).to(device)
    ckpt = Path("experiments/svg_assembly/reports/gnn_supervised_best.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("No checkpoint found, using random weights")

    model.eval()

    # 1. Check GT tree structure statistics
    print("\n=== GT Tree Structure ===")
    depths = []
    widths = []
    n_internal = []
    for rec in test_records:
        gt = build_tree_from_list(rec.assembly_tree)
        nodes = nonleaf_nodes(gt)
        n_internal.append(len(nodes))
        # depth
        def tree_depth(node):
            if not node.children:
                return 0
            return 1 + max(tree_depth(c) for c in node.children)
        depths.append(tree_depth(gt))
        # max children per node
        def max_children(node):
            mc = len(node.children)
            for c in node.children:
                mc = max(mc, max_children(c))
            return mc
        widths.append(max_children(gt))

    print(f"  Parts per object: min={min(r.num_parts for r in test_records)}, "
          f"max={max(r.num_parts for r in test_records)}, "
          f"mean={sum(r.num_parts for r in test_records)/len(test_records):.1f}")
    print(f"  Internal nodes: min={min(n_internal)}, max={max(n_internal)}, "
          f"mean={sum(n_internal)/len(n_internal):.1f}")
    print(f"  Tree depth: min={min(depths)}, max={max(depths)}, "
          f"mean={sum(depths)/len(depths):.1f}")
    print(f"  Max children per node: min={min(widths)}, max={max(widths)}, "
          f"mean={sum(widths)/len(widths):.1f}")

    # 2. Check what greedy decoder produces
    print("\n=== Greedy Decoder Output ===")
    for i, rec in enumerate(test_records[:5]):
        pred = greedy_decode(model, rec, device)
        gt_tree = build_tree_from_list(rec.assembly_tree)
        pred_tree = build_tree_from_list(pred)
        m = eval_tree(gt_tree, pred_tree)
        print(f"\n  Object {i}: {rec.category}/{rec.name} ({rec.num_parts} parts)")
        print(f"    GT tree:   {rec.assembly_tree}")
        print(f"    Pred tree: {pred}")
        print(f"    GT nonleaf:   {[n.parts for n in nonleaf_nodes(gt_tree)]}")
        print(f"    Pred nonleaf: {[n.parts for n in nonleaf_nodes(pred_tree)]}")
        print(f"    Simple F1={m['simple']['f1']:.3f}  Hard F1={m['hard']['f1']:.3f}")

    # 2b. Group-aware greedy decoder
    print("\n=== Group-Aware Greedy Decoder Output ===")
    for i, rec in enumerate(test_records[:5]):
        pred = group_greedy_decode(model, rec, device, threshold=0.0)
        gt_tree = build_tree_from_list(rec.assembly_tree)
        pred_tree = build_tree_from_list(pred)
        m = eval_tree(gt_tree, pred_tree)
        print(f"\n  Object {i}: {rec.category}/{rec.name} ({rec.num_parts} parts)")
        print(f"    GT tree:   {rec.assembly_tree}")
        print(f"    Pred tree: {pred}")
        print(f"    GT nonleaf:   {[n.parts for n in nonleaf_nodes(gt_tree)]}")
        print(f"    Pred nonleaf: {[n.parts for n in nonleaf_nodes(pred_tree)]}")
        print(f"    Simple F1={m['simple']['f1']:.3f}  Hard F1={m['hard']['f1']:.3f}")

    # 3. Check pair scores distribution
    print("\n=== Pair Score Distribution (first object) ===")
    rec = test_records[0]
    part_feats = torch.tensor(rec.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(rec.graph.edge_index, dtype=torch.long, device=device)
    part_embeds = model.encode_parts(part_feats, edge_index)

    # Initial state: all singletons
    clusters = [frozenset([i]) for i in range(rec.num_parts)]
    pairs = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))

    logits = model.score_pairs_batch(
        part_embeds, pairs,
        [list(sorted(c)) for c in clusters],
    )
    probs = torch.sigmoid(logits)
    print(f"  {len(pairs)} pairs, logits: min={logits.min():.3f} max={logits.max():.3f} "
          f"mean={logits.mean():.3f} std={logits.std():.3f}")
    print(f"  probs:  min={probs.min():.3f} max={probs.max():.3f} "
          f"mean={probs.mean():.3f} std={probs.std():.3f}")

    # Top 5 pairs
    top5 = probs.argsort(descending=True)[:5]
    print(f"  Top 5 pairs by prob:")
    for idx in top5:
        ca, cb = pairs[idx]
        print(f"    ({ca}, {cb}) -> logit={logits[idx]:.3f} prob={probs[idx]:.3f}")

    # 4. Check if Simple=0.396 is just from trivial matches
    print("\n=== Trivial Match Analysis ===")
    simple_from_trivial = 0
    total_simple = 0
    for rec in test_records:
        gt_tree = build_tree_from_list(rec.assembly_tree)
        gt_nodes = nonleaf_nodes(gt_tree)
        pred = greedy_decode(model, rec, device)
        pred_tree = build_tree_from_list(pred)
        pred_nodes = nonleaf_nodes(pred_tree)
        for gt in gt_nodes:
            total_simple += 1
            for pred in pred_nodes:
                if gt.parts == pred.parts:
                    simple_from_trivial += 1
                    break
    print(f"  Simple matches: {simple_from_trivial}/{total_simple} = "
          f"{simple_from_trivial/max(total_simple,1):.3f}")


if __name__ == "__main__":
    main()
