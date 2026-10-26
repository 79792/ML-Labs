"""Hand-calculated params and FLOPs (Ch. 4 formulas), per single image."""

# formulas
def linear_flops(n_in, n_out, seq=1):
    return 2 * seq * n_in * n_out


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def attn_layer_flops(seq, d, ff):
    qkv = 2 * seq * d * 3 * d        # Q, K, V projections
    scores = 2 * seq * seq * d       # Q @ K^T
    weighted = 2 * seq * seq * d     # softmax(..) @ V
    out = 2 * seq * d * d            # output projection
    ffn = 2 * seq * d * ff * 2       # two FFN linears
    return {"qkv": qkv, "scores": scores, "attn_v": weighted,
            "out_proj": out, "ffn": ffn,
            "total": qkv + scores + weighted + out + ffn}


def hand_rows():
    """One representative layer of each type + whole-model totals."""
    a = attn_layer_flops(16, 64, 128)
    layers = [
        {"model": "MLP", "layer": "Linear 784->256", "hand_flops": linear_flops(784, 256)},
        {"model": "CNN", "layer": "Conv2d 16->32, 3x3, out 14x14",
         "hand_flops": conv_flops(3, 16, 32, 14, 14)},
        {"model": "Attention", "layer": "1 encoder layer (seq=16, d=64, ff=128)",
         "hand_flops": a["total"]},
    ]
    totals = {
        "MLP": linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10),
        "CNN": conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14)
               + linear_flops(32 * 7 * 7, 10),
        "Attention": linear_flops(49, 64, seq=16) + 2 * a["total"] + linear_flops(64, 10),
    }
    return layers, totals, a


if __name__ == "__main__":
    layers, totals, a = hand_rows()
    for r in layers:
        print(f"{r['model']:<10} {r['layer']:<40} {r['hand_flops']:>12,}")
    print("\nAttention layer breakdown:", {k: f"{v:,}" for k, v in a.items()})
    print("\nWhole-model totals:", {k: f"{v:,}" for k, v in totals.items()})
