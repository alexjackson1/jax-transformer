from debug_utils import print_nested_structure


if __name__ == "__main__":
    from tx import HookedTransformer, ModelConfig
    import gpt2

    config = ModelConfig(
        vocab_dim=50257,
        model_dim=768,
        context_length=1024,
        num_layers=12,
        num_heads=12,
        head_dim=64,
        mlp_dim=3072,
        layer_norm_eps=1e-5,
    )
    model = HookedTransformer.from_config(config=config)

    input_text = "Hello, "
    print(input_text, flush=True)
    tokens = gpt2.tokenizer(input_text, return_tensors="jax")["input_ids"]
    output = model.apply({"params": gpt2.GPT2_PARAMS}, tokens)
    print(output.shape, flush=True)
