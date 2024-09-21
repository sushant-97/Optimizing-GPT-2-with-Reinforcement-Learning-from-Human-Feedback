# Reproducing GPT2 Model

Strucutre
```python
GPT
    __init__
        TransformerDecoder
        lm_head
    forward(x)
        return lm_head(TransformerDecoder(x))
    from_checkpoint
    from_pretrained
    generate
    batch_generate
```
```python
TransformerDecoder
    __init__()
        # Layers
            # token_embedding_layer
            # postion_embedding_layer
            # input_dropout
            # decoder_blocks = [TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)]
            # ln = LayerNorm
        
    forward(x, attention_mask):
        # pos, token_embeddings, pos_embeddings
        x = input_dropout(token_embeddings + pos_embeddings)

        # Stack N blocks
        for block in self.decoder_blocks:
            x = block(x)

        y = ln(x)

    TransformerDecoderBlock
        __init__()
            # Layers
                # ln1, ln2 = LayerNorm
                # mmsa = MaskedMultiheadSelfAttention
                # ffn = FeedForwardNetworks
        forward(x, attention_mask):
            # identity1 = x     # For Residual connection
            # x = identity1 + mmsa(ln1(x))
            # identity2 = x
            # x = identity2 + mmsa(ln1(x))
            # return x

    MaskedMultiheadSelfAttention
        __init__
            Q, K, V, attention_dropout, output_dropout, mask
        forward(x, attention_mask):
            # x3 <- qkv_projection(x)
            # Q, K, V = x3.split()
            # attention = Q @ K.transpose(2, 3)
            # apply attention_mask to attention
            # attention_dropout
            # weighted_value = attention @ V
            # project weighted_value linearly to get output
            # y = output_projection(weighted_value)



```


[base_implementation](https://github.com/karpathy/nanoGPT/blob/master/model.py)

[reference2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

