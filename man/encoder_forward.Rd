% tinyrox says don't edit this manually, but it can't stop you!
\name{encoder_forward}
\alias{encoder_forward}
\title{Fused Encoder Forward}
\usage{
encoder_forward(mel, global_weights, layer_weights, n_head, n_ctx)
}
\arguments{
\item{mel}{Input tensor (batch, n_mels, n_frames) mel spectrogram.}

\item{global_weights}{List of 7 tensors: conv1.weight, conv1.bias,
conv2.weight, conv2.bias, positional_embedding, ln_post.weight, ln_post.bias.}

\item{layer_weights}{List of N lists, each containing 15 weight tensors
(same order as \code{\link{transformer_encoder_layer}}).}

\item{n_head}{Integer number of attention heads.}

\item{n_ctx}{Integer maximum context length for sequence truncation.}
}
\value{
Output tensor (batch, seq_len, n_state).
}
\description{
Executes the entire encoder forward pass in a single C++ call:
conv1+gelu, conv2+gelu, permute, positional embedding, all N
encoder layers (self-attention + MLP), and final layer norm.
}
