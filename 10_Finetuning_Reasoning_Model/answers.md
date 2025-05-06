### Completed Notebook

https://colab.research.google.com/drive/1A0OI8-ynsnUxOAiqErFK_kD-Jk5WLco4?usp=sharing

---

### ❓Question #1:

What exactly is happening in the double quantization step?

> NOTE: You can use the paper provided to find the answer!

The double quantization step is a method to compress the weights of LLMs efficiently while preserving performance. It saves memory by compressing not just the model weights, but also the metadata used for their quantization.

Double quantization involves two layers of quantization: the first step where groups of weights are quantized, each with their own scale and zero-point, and the second step where the quantization constants (scale and zero-point) themselves are quantized.

---

### ❓Question #2:

Label the image with the appropriate layer from `meta-llama/Llama-3.2-3B-Instruct`'s architecture.

- EXAMPLE - Layer Norm:

  - `(input_layernorm): LlamaRMSNorm()`
  - `(post_attention_layernorm): LlamaRMSNorm()`
  - `(norm): LlamaRMSNorm()`

- Feed Forward:

  - `(mlp) LlamaMLP()`

- Masked Multi Self-Attention:

  - `(self_attn): LlamaAttention()`

- Text & Position Embed:

  - `(embed_tokens): Embedding(128256, 3072, padding_idx=128004)`

- Text Prediction:

  - `(lm_head): Linear(in_features=3072, out_features=128256, bias=False)`

---

### ❓Question #3:

What, in your own words, is LoRA doing?

LoRA is adapting the LLM to a specific task by injecting trainable adapters into specific layers of the model instead of modifying the model's weights.

---

### ❓Question #4:

Describe what the following parameters are doing:

- `warmup_ratio`
- `learning_rate`
- `lr_scheduler_type`

> NOTE: Feel free to consult the [documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) or other resources!

**learning_rate**: This parameter determines how big each step is when updating the weights during training. A high learning rate might lead to faster progress but risks overshooting the optimal solution, while a low learning rate, though more stable, can result in slow convergence.

**warmup_ratio**: This parameter defines what fraction of total training steps will be used for the learning rate warm-up phase. In the warm-up phase, the learning rate gradually increases from 0 to the target `learning_rate`, this way we ensure that early training is stable by preventing sudden large weight updates at the start. If `max_steps` is 170 and `warmup_ratio` is 0.1, then this means that the first 17 steps will be warm-up steps.

**lr_scheduler_type**: This parameter controls how the `learning_rate` decays or adjusts over time during training. For example, `lr_scheduler_type="cosine"` means that the learning rate will gradually decrease following a cosine curve from the initial value down toward zero, this tends to fine-tune weights and converge more smoothly than using a linear decay.
