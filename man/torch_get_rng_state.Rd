% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_get_rng_state}
\alias{torch_get_rng_state}
\alias{torch_set_rng_state}
\title{Get/set RNG state (stubs)}
\usage{
torch_get_rng_state()

torch_set_rng_state(state)
}
\arguments{
\item{state}{A raw vector of RNG state (from torch_get_rng_state).}
}
\value{
For get, the current RNG state as a tensor. For set, invisible NULL.
}
\description{
Get/set RNG state (stubs)
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}

\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
