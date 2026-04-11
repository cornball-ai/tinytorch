% tinyrox says don't edit this manually, but it can't stop you!
\name{transformer_encoder_layer}
\alias{transformer_encoder_layer}
\title{Fused Transformer Encoder Layer}
\usage{
transformer_encoder_layer(x, weights, n_head)
}
\arguments{
\item{x}{Input tensor (batch, seq_len, n_state).}

\item{weights}{List of 15 weight tensors in fixed order (see details).}

\item{n_head}{Integer number of attention heads.}
}
\value{
Output tensor (batch, seq_len, n_state).
}
\description{
Executes one pre-norm encoder layer entirely in C++.
Replaces ~20 R-to-C++ crossings with a single call.
}
\details{
Weight list order (15 tensors):
\enumerate{
  \item attn_ln.weight, attn_ln.bias
  \item self_q.weight, self_q.bias
  \item self_k.weight (no bias)
  \item self_v.weight, self_v.bias
  \item self_out.weight, self_out.bias
  \item mlp_ln.weight, mlp_ln.bias
  \item fc1.weight, fc1.bias
  \item fc2.weight, fc2.bias
}
}
\examples{
\donttest{
if (torch_is_installed()) {
# Low-level transformer op; see encoder_forward() for full pipeline
}
}
}
