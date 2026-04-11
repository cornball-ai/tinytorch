% tinyrox says don't edit this manually, but it can't stop you!
\name{greedy_decode}
\alias{greedy_decode}
\title{Greedy Decode}
\usage{
greedy_decode(initial_tokens, global_weights, layer_weights, cross_cache_k,
              cross_cache_v, n_head, max_length, eot_token)
}
\arguments{
\item{initial_tokens}{Integer vector of initial token IDs (0-indexed).}

\item{global_weights}{List of 4 tensors: token_emb.weight, pos_emb.weight,
final_ln.weight, final_ln.bias.}

\item{layer_weights}{List of N lists, each containing 21 weight tensors.}

\item{cross_cache_k}{List of N pre-computed cross-attention key tensors.}

\item{cross_cache_v}{List of N pre-computed cross-attention value tensors.}

\item{n_head}{Integer number of attention heads.}

\item{max_length}{Maximum output sequence length.}

\item{eot_token}{End-of-text token ID (0-indexed).}
}
\value{
Integer vector of generated token IDs (0-indexed).
}
\description{
Runs the entire autoregressive decode loop in C++.
One boundary crossing for the full sequence -- no per-token R overhead.
}
\examples{
\donttest{
if (torch_is_installed()) {
# Specialized Whisper greedy decoder; see package vignettes
}
}
}
