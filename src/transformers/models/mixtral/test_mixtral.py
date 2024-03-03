from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.models.mixtral import MixtralPreTrainedModel, MixtralForCausalLM, MixtralConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

config = MixtralConfig(vocab_size=32000,
        hidden_size=32,
        intermediate_size=14336,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=2,
        output_router_logits=True,
        router_aux_loss_coef=0.001)

model = MixtralForCausalLM(config)

print(model)

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")
print (inputs)

outputs = model.generate(**inputs, max_new_tokens=20, output_router_logits=True)
print (outputs)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))