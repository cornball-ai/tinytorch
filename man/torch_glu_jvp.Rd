% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_glu_jvp}
\alias{torch_glu_jvp}
\title{Glu jvp}
\usage{
torch_glu_jvp(glu, x, dx, dim)
}
\arguments{
\item{glu}{A torch_tensor.}

\item{x}{A torch_tensor.}

\item{dx}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::glu_jvp}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
