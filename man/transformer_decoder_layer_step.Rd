% tinyrox says don't edit this manually, but it can't stop you!
\name{transformer_decoder_layer_step}
\alias{transformer_decoder_layer_step}
\title{Fused Transformer Decoder Layer Step}
\usage{
transformer_decoder_layer_step(x, weights, self_cache_k, self_cache_v,
                               cross_cache_k, cross_cache_v, n_head)
}
\arguments{
\item{x}{Input tensor (batch, 1, n_state).}

\item{weights}{List of 21 weight tensors in fixed order (see details).}

\item{self_cache_k}{Self-attention key cache (batch, n_head, seq_so_far, head_dim).}

\item{self_cache_v}{Self-attention value cache.}

\item{cross_cache_k}{Cross-attention key cache (batch, n_head, src_len, head_dim).}

\item{cross_cache_v}{Cross-attention value cache.}

\item{n_head}{Integer number of attention heads.}
}
\value{
Named list: output (batch, 1, n_state), self_k (updated), self_v (updated).
}
\description{
Executes one pre-norm decoder layer entirely in C++ for incremental decoding
(seq_len=1 with KV cache). Replaces ~26 R-to-C++ crossings with a single call.
}
\details{
Weight list order (21 tensors):
\enumerate{
  \item attn_ln.weight, attn_ln.bias
  \item self_q.weight, self_q.bias
  \item self_k.weight (no bias)
  \item self_v.weight, self_v.bias
  \item self_out.weight, self_out.bias
  \item cross_attn_ln.weight, cross_attn_ln.bias
  \item cross_q.weight, cross_q.bias
  \item cross_out.weight, cross_out.bias
  \item mlp_ln.weight, mlp_ln.bias
  \item fc1.weight, fc1.bias
  \item fc2.weight, fc2.bias
}
}
