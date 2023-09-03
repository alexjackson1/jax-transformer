from transformers import GPT2TokenizerFast, FlaxGPT2LMHeadModel, GenerationConfig


class GPT2(FlaxGPT2LMHeadModel):
    @property
    def vocab_size(self):
        return self._params["transformer"]["wte"]["embedding"].shape[0]

    @property
    def transformer(self):
        return self._params["transformer"]

    @property
    def ln_f(self):
        return self._params["transformer"]["ln_f"]

    @property
    def embed(self):
        return self._params["transformer"]["wte"]

    @property
    def pos_embed(self):
        return self._params["transformer"]["wpe"]

    @property
    def blocks(self):
        d = self._params["transformer"]["h"]
        return [d[k] for k in sorted(d, key=int)]

    def block_dict(self, block):
        return {
            "LayerNorm_0": block["ln_1"],
            # "Attention_0": {
            #     "kernel_query": block["attn"]["c_attn"]["w"],
            #     "bias_query": ,
            #     "kernel_key": "",
            #     "bias_key": "",
            #     "kernel_value": "",
            #     "bias_value": "",
            #     "kernel_out": "",
            #     "bias_out": "",
            # },
            "LayerNorm_1": block["ln_2"],
            "MLP_0": {
                "DenseGeneral_0": block["mlp"]["c_fc"],
                "DenseGeneral_1": block["mlp"]["c_proj"],
            },
        }

    def params_dict(self):
        params = {"Embed_0": self.embed, "PosEmbed_0": self.pos_embed}
        for i, block in enumerate(self.blocks):
            block_dict = self.block_dict(block)
            params[f"TransformerBlock_{i}"] = block_dict

        params[f"LayerNorm_0"] = self.ln_f
        params["Unembed_0"] = jnp.transpose(self.embed["embedding"])
        return {"params": params}


tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2")

if __name__ == "__main__":
    import jax.numpy as jnp
    from debug_utils import print_nested_structure
    from tx.modules import LayerNorm

    # # text = "Hello, my name is "
    # # tokens = tokenizer(text, return_tensors="jax")
    # # print("Tokens:", tokens["input_ids"].shape)
    # # config = GenerationConfig(max_length=50, pad_token_id=tokenizer.eos_token_id)
    # # output = model.generate(tokens["input_ids"], config)
    # # print(tokenizer.batch_decode(output["sequences"]))

    # text = "Hello, my name is "
    # tokens = tokenizer(text, return_tensors="jax")
    # print("Tokens:", tokens["input_ids"].shape)
    # output = model(
    #     tokens["input_ids"],
    #     return_dict=True,
    #     output_hidden_states=True,
    #     output_attentions=True,
    # )

    # # print_nested_structure(output)
    # print(output["hidden_states"][-1].shape)

    # # print(type(model.ln_f))
    # # input = jnp.ones((2, 4, 768))
    # # LayerNorm().apply({"params": model.ln_f}, input)
    print_nested_structure(model.blocks[0])
    # print_nested_structure(model.params_dict(), max_depth=3)
    # print_nested_structure(model.params_dict())
    # model
