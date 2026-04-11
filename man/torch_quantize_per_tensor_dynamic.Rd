% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantize_per_tensor_dynamic}
\alias{torch_quantize_per_tensor_dynamic}
\title{Quantize per tensor dynamic}
\usage{
torch_quantize_per_tensor_dynamic(self, dtype, reduce_range)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dtype}{A torch dtype (e.g., torch_float32).}

\item{reduce_range}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantize_per_tensor_dynamic}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
