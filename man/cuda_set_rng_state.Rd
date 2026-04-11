% tinyrox says don't edit this manually, but it can't stop you!
\name{cuda_set_rng_state}
\alias{cuda_set_rng_state}
\title{Cuda set rng state}
\usage{
cuda_set_rng_state(state, device = 0L)
}
\arguments{
\item{state}{RNG state.}

\item{device}{Parameter passed to the underlying ATen operator.}
}
\description{
Cuda set rng state
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
