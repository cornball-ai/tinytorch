% tinyrox says don't edit this manually, but it can't stop you!
\name{decoder_forward_step}
\alias{decoder_forward_step}
\title{Fused Decoder Forward Step}
\usage{
decoder_forward_step(token_ids, global_weights, layer_weights, self_cache_k,
                     self_cache_v, cross_cache_k, cross_cache_v, n_head, offset)
}
\arguments{
\item{token_ids}{Integer tensor (batch, seq_len) of 0-indexed token IDs.}

\item{global_weights}{List of 4 tensors: token_emb.weight, pos_emb.weight,
final_ln.weight, final_ln.bias.}

\item{layer_weights}{List of N lists, each containing 21 weight tensors
(same order as \code{\link{transformer_decoder_layer_step}}).}

\item{self_cache_k}{List of N tensors for self-attention key cache,
or NULL for prefill.}

\item{self_cache_v}{List of N tensors for self-attention value cache,
or NULL for prefill.}

\item{cross_cache_k}{List of N pre-computed cross-attention key tensors.}

\item{cross_cache_v}{List of N pre-computed cross-attention value tensors.}

\item{n_head}{Integer number of attention heads.}

\item{offset}{Integer position offset for positional embedding.}
}
\value{
Named list: token_id (integer), self_cache_k, self_cache_v,
  cross_cache_k, cross_cache_v.
}
\description{
Executes the entire decoder forward pass in a single C++ call:
token embedding, positional embedding, all N decoder layers
(self-attn + cross-attn + MLP with KV cache), final layer norm,
logits projection, and argmax.
}
\examples{
\donttest{
if (torch_is_installed()) {
# Specialized Whisper decoder step; see package vignettes
}
}
}
