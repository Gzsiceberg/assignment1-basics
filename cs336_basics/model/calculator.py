from cs336_basics.logger import print_and_log


def calc_num_params(
    vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, ffn_type: str = "swiglu"
) -> int:
    embedding_params = vocab_size * d_model
    mha_params = 4 * d_model * d_model * num_layers
    ffn_params = 3 if ffn_type == "swiglu" else 2 * d_model * d_ff * num_layers
    rmsnorm_params = 2 * d_model * num_layers
    block_params = mha_params + ffn_params + rmsnorm_params
    rmsnorm_params = d_model
    lmhead_params = d_model * vocab_size
    total_params = embedding_params + block_params + rmsnorm_params + lmhead_params
    print_and_log(f"Embedding params: {embedding_params:,} Percent: {embedding_params/total_params:.2%}")
    print_and_log(f"MHA params: {mha_params:,} Percent: {mha_params/total_params:.2%}")
    print_and_log(f"FFN params: {ffn_params:,} Percent: {ffn_params/total_params:.2%}")
    print_and_log(f"rmsnorm params: {rmsnorm_params:,} Percent: {rmsnorm_params/total_params:.2%}")
    print_and_log(f"Block params: {block_params:,} Percent: {block_params/total_params:.2%}")
    print_and_log(f"RMSNorm params: {rmsnorm_params:,} Percent: {rmsnorm_params/total_params:.2%}")
    print_and_log(f"LM Head params: {lmhead_params:,} Percent: {lmhead_params/total_params:.2%}")
    print_and_log(f"Total params: {total_params:,} ({total_params * 4 / (1024**3):.2f} GB)")
    return total_params


def calc_flops(
    seq_len: int, batch_size: int, num_layers: int, d_model: int, d_ff: int, vocab_size: int, ffn_type: str = "swiglu"
) -> int:
    atten_flops = num_layers * (4 * seq_len * seq_len * d_model + 8 * d_model * d_model * seq_len) * batch_size
    ffn_flops = num_layers * (6 if ffn_type == "ffn_type" else 4 * d_model * d_ff * seq_len) * batch_size
    flops_block = atten_flops + ffn_flops
    flops_out = 2 * d_model * vocab_size * seq_len * batch_size
    total_flops = flops_block + flops_out
    print_and_log(
        f"FLOPS for attention: {atten_flops / 1e12:.2f} TFLOPS {atten_flops:,} FLOPS Percent: {atten_flops/total_flops:.2%}"
    )
    print_and_log(f"FLOPS for FFN: {ffn_flops / 1e12:.2f} TFLOPS {ffn_flops:,} FLOPS, Percent: {ffn_flops/total_flops:.2%}")
    print_and_log(
        f"FLOPS per Transformer block: {flops_block / 1e12:.2f} TFLOPS {flops_block:,} FLOPS, Percent: {flops_block/total_flops:.2%}"
    )
    print_and_log(
        f"FLOPS for output layer: {flops_out / 1e12:.2f} TFLOPS {flops_out:,} FLOPS, Percent: {flops_out/total_flops:.2%}"
    )
    print_and_log(f"Total FLOPS: {total_flops / 1e12:.2f} TFLOPS {total_flops:,} FLOPS")
    return total_flops


def calc_llm_memory(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    batch_size: int,
    ffn_type="swiglu",
) -> None:
    d_ff = 4 * d_model
    embedding_params = vocab_size * d_model
    embedding_gradients = embedding_params

    rmsnorm_params = 2 * d_model * num_layers
    rmsnorm_params_gradients = rmsnorm_params
    rmsnorm_activation = 2 * context_length * d_model * num_layers

    mha_params = 4 * d_model * d_model * num_layers
    mha_gradients = mha_params
    mha_qkv_activations = 3 * context_length * d_model * num_layers
    mha_attention_scores_activations = num_heads * context_length * context_length * num_layers
    mha_softmax_activations = num_heads * context_length * context_length * num_layers
    mha_attention_values_activations = context_length * d_model * num_layers
    mha_o_activations = context_length * d_model * num_layers
    mha_activations = (
        mha_qkv_activations
        + mha_o_activations
        + mha_attention_scores_activations
        + mha_softmax_activations
        + mha_attention_values_activations
    )

    ffn_params = 3 if ffn_type == "swiglu" else 2 * d_model * d_ff * num_layers
    ffn_params_gradients = ffn_params
    ffn_silu_activation = context_length * d_ff * num_layers
    ffn_fc1_activation = context_length * d_ff * num_layers
    ffn_fc3_activation = context_length * d_ff * num_layers if ffn_type == "swiglu" else 0
    ffn_fc2_activation = context_length * d_model * num_layers
    ffn_activation = ffn_silu_activation + ffn_fc1_activation + ffn_fc2_activation + ffn_fc3_activation

    block_params = mha_params + ffn_params + rmsnorm_params
    block_activations = mha_activations + ffn_activation + rmsnorm_activation

    lmfinal_rmsnorm_params = d_model
    lmfinal_rmsnorm_params_gradients = lmfinal_rmsnorm_params
    lmfinal_rmsnorm_activation = context_length * d_model

    lmhead_params = d_model * vocab_size
    lmhead_params_gradients = lmhead_params
    lmhead_params_activations = context_length * vocab_size

    cross_entropy_loss_activation = context_length

    total_params = embedding_params + block_params + lmfinal_rmsnorm_params + lmhead_params
    total_gradients = (
        embedding_gradients
        + mha_gradients
        + ffn_params_gradients
        + rmsnorm_params_gradients
        + lmfinal_rmsnorm_params_gradients
        + lmhead_params_gradients
    )
    total_adammw_states = 2 * total_params
    total_activations = (
        block_activations + lmfinal_rmsnorm_activation + lmhead_params_activations + cross_entropy_loss_activation
    )

    print_and_log(f"Total parameters: {total_params:,}")
    print_and_log(f"Total activations: {total_activations:,}")
    print_and_log(f"params + gradients + states: {(total_params + total_gradients + total_adammw_states) * 4:,}")
    print_and_log(f"activations per batch size: {total_activations * 4:,}")
    params_memory = total_params * 4 / (1024**3)
    gradients_memory = total_gradients * 4 / (1024**3)
    adammw_states_memory = total_adammw_states * 4 / (1024**3)
    activations_memory = batch_size * total_activations * 4 / (1024**3)
    total_memory = (total_params + total_gradients + total_activations) * 4 / (1024**3)
    print_and_log(f"Parameters memory (GB): {params_memory:.2f}")
    print_and_log(f"Gradients memory (GB): {gradients_memory:.2f}")
    print_and_log(f"AdamW states memory (GB): {adammw_states_memory:.2f}")
    print_and_log(f"Activations memory (GB): {activations_memory:.2f}")
    print_and_log(f"Total memory (GB): {total_memory:.2f}")


def test():
    print_and_log("--- GPT-XL-like Model Configuration ---")
    vocab_size = 50_257
    num_layers = 48
    d_model = 1600
    num_heads = 24
    d_ff = 6400
    max_seq_len = 1024
    theta = 100000.0
    ffn_type = "silu"
    # model = TransformerLM(
    #     vocab_size=vocab_size,
    #     num_layers=num_layers,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     d_ff=d_ff,
    #     max_seq_len=max_seq_len,
    #     theta=theta,
    # )
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Model parameters: {total_params:,}")
    print_and_log("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff, ffn_type)

    print_and_log("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len,
        batch_size=1,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        ffn_type=ffn_type,
    )
    print_and_log("-" * 80)

    print_and_log("--- GPT-2 small Model Configuration ---")
    num_layers = 12
    d_model = 768
    num_heads = 12
    print_and_log("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff, ffn_type)

    print_and_log("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len,
        batch_size=1,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        ffn_type=ffn_type,
    )
    print_and_log("-" * 80)

    print_and_log("--- GPT-2 medium Model Configuration ---")
    num_layers = 24
    d_model = 1024
    num_heads = 16
    print_and_log("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff, ffn_type)

    print_and_log("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len,
        batch_size=1,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        ffn_type=ffn_type,
    )
    print_and_log("-" * 80)

    print_and_log("--- GPT-2 large Model Configuration ---")
    num_layers = 36
    d_model = 1280
    num_heads = 20
    print_and_log("--- Estimating Parameters ---")
    estimate_params = calc_num_params(vocab_size, num_layers, d_model, num_heads, d_ff, ffn_type)

    print_and_log("\n--- Estimating FLOPS ---")
    flops = calc_flops(
        seq_len=max_seq_len,
        batch_size=1,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        ffn_type=ffn_type,
    )
    print_and_log("-" * 80)

    max_seq_len = 16_384
    print_and_log("--- Estimating FLOPS for 16K context ---")
    flops = calc_flops(
        seq_len=max_seq_len,
        batch_size=1,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        ffn_type=ffn_type,
    )
    print_and_log("-" * 80)


if __name__ == "__main__":
    vocab_size = 50_257
    num_layers = 48
    d_model = 1600
    num_heads = 24
    d_ff = 6400
    max_seq_len = 1024
    theta = 100000.0
    batch_size = 1
    ffn_type = "silu"

    P = 2 * vocab_size * d_model + num_layers * (12 * d_model * d_model + 2 * d_model) + d_model
    print_and_log(f"GPT-2 XL model parameters using SiLU: {P:,}")
    memorgy = P * 4 * 3
    print_and_log(f"GPT-2 XL model parameters memory bytes: {memorgy:,}")

    context_length = max_seq_len
    block_C = num_layers * (16 * context_length * d_model + 2 * num_heads * context_length * context_length)
    print_and_log(f"GPT-2 XL model block activations {block_C:,}")
    final_norm_c = context_length * d_model
    lm_head_c = context_length * vocab_size
    loss_c = context_length
    total_C = block_C + final_norm_c + lm_head_c + loss_c
    print_and_log(f"GPT-2 XL model total activations {total_C:,}")

    print_and_log("-" * 80)
    print_and_log(f"GPT-2 XL model memory usage for batch size {batch_size}:")
    calc_llm_memory(vocab_size, max_seq_len, num_layers, d_model, num_heads, batch_size, ffn_type=ffn_type)

    print_and_log("-" * 80)
    print_and_log(f"GPT-2 XL model memory usage for batch size {batch_size} and 16K context:")
    max_seq_len = 16_384
    calc_llm_memory(vocab_size, max_seq_len, num_layers, d_model, num_heads, batch_size, ffn_type=ffn_type)

    print_and_log("-" * 80)
    batch_size = 1024
    step = 400_000
    forward_flops = 2 * context_length * batch_size * P
    backward_flops = 2 * forward_flops
    peak_flops = 19.5e12
    mfu = 0.5
    total_flops = (forward_flops + backward_flops) * step
    print_and_log(f"total flops: {total_flops/1e12:,.2f} TFLOPS")
    t_total = total_flops / (peak_flops * mfu)
    print_and_log(f"time per iteration (seconds): {t_total:,.2f} days: {t_total/86400:.2f}")
