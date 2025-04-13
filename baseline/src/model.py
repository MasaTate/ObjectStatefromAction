import torch
from clip.model import Transformer
from torch import nn
import math


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ClipTextEncoder(torch.nn.Module):
    def __init__(self, params, train_backbone=True):
        super(ClipTextEncoder, self).__init__()

        self.args = params
        """
        for k, v in params.items():
            print(k)
        """
        self.embed_dim = params["text_projection"].shape[1]
        self.context_length = params["positional_embedding"].shape[0]
        self.vocab_size = params["token_embedding.weight"].shape[0]
        self.transformer_width = params["ln_final.weight"].shape[0]
        self.transformer_heads = self.transformer_width // 64
        self.transformer_layers = len(
            set(
                k.split(".")[2] for k in params if k.startswith("transformer.resblocks")
            )
        )
        self.dtype = params["transformer.resblocks.0.ln_1.weight"].dtype
        print(self.dtype)

        self.args = dict(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, self.transformer_width)
        )
        self.ln_final = LayerNorm(self.transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(self.transformer_width, self.embed_dim)
        )

        self.transformer = Transformer(**self.args)

        if "transformer.resblocks.0.ln_1.weight" in params:
            print("loading state dict for text encoder...")
            self.transformer.load_state_dict(
                {
                    k[len("transformer.") :]: v
                    for k, v in params.items()
                    if k.startswith("transformer.")
                }
            )

            self.token_embedding.load_state_dict(
                {
                    k[len("token_embedding.") :]: v
                    for k, v in params.items()
                    if k.startswith("token_embedding.")
                }
            )

            self.positional_embedding.data.copy_(
                params["positional_embedding"].data[: self.context_length]
            )

            self.ln_final.load_state_dict(
                {
                    k[len("ln_final.") :]: v
                    for k, v in params.items()
                    if k.startswith("ln_final.")
                }
            )

            self.text_projection.data.copy_(params["text_projection"].data)

        else:
            raise ValueError("No text encoder weights found in state dict.")

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
