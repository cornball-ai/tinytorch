% tinyrox says don't edit this manually, but it can't stop you!
\name{prepare_cross_caches}
\alias{prepare_cross_caches}
\title{Prepare Cross-Attention Caches}
\usage{
prepare_cross_caches(encoder_output, cross_kv_weights, n_head)
}
\arguments{
\item{encoder_output}{Encoder output tensor (batch, src_len, n_state).}

\item{cross_kv_weights}{List of N lists, each containing 3 tensors:
cross_k.weight, cross_v.weight, cross_v.bias.}

\item{n_head}{Integer number of attention heads.}
}
\value{
Named list with k and v, each a list of N tensors
  shaped (batch, n_head, src_len, head_dim).
}
\description{
Projects encoder output through each decoder layer's cross-attention
K/V weights in a single C++ call.
}
