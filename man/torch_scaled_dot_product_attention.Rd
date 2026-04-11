% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scaled_dot_product_attention}
\alias{torch_scaled_dot_product_attention}
\title{Scaled Dot-Product Attention}
\usage{
torch_scaled_dot_product_attention(query, key, value, attn_mask = NULL,
                                   dropout_p = 0, is_causal = FALSE)
}
\arguments{
\item{query}{Query tensor (batch, heads, seq_q, head_dim).}

\item{key}{Key tensor (batch, heads, seq_k, head_dim).}

\item{value}{Value tensor (batch, heads, seq_k, head_dim).}

\item{attn_mask}{Optional attention mask tensor.}

\item{dropout_p}{Dropout probability (default 0).}

\item{is_causal}{Whether to apply causal masking (default FALSE).}
}
\value{
Output tensor.
}
\description{
Scaled Dot-Product Attention
}
\examples{
\donttest{
if (torch_is_installed()) {
if (torch_is_installed()) {
  # Basic usage
  query <- torch_randn(2, 8, 10, 64)  # (batch, heads, seq_len, dim)
  key <- torch_randn(2, 8, 10, 64)
  value <- torch_randn(2, 8, 10, 64)

  output <- torch_scaled_dot_product_attention(query, key, value)

  # With causal masking (for autoregressive models)
  output <- torch_scaled_dot_product_attention(
    query, key, value,
    is_causal = TRUE
  )

  # With attention mask
  seq_len <- 10
  attn_mask <- torch_ones(seq_len, seq_len)
  attn_mask <- torch_tril(attn_mask)  # Lower triangular mask
  output <- torch_scaled_dot_product_attention(
    query, key, value,
    attn_mask = attn_mask
  )
}

}
}
}
