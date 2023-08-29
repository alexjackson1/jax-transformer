from transformers import GPT2TokenizerFast, FlaxGPT2Model

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
reference_gpt2 = FlaxGPT2Model.from_pretrained("gpt2")


if __name__ == "__main__":
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = tokenizer(reference_text, return_tensors="jax")
    print(tokens)
    print(tokens["input_ids"].shape)
    print("Input text:", reference_text)
    print(reference_gpt2(tokens["input_ids"]))
