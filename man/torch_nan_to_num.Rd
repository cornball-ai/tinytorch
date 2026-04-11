% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nan_to_num}
\alias{torch_nan_to_num}
\title{Nan to num}
\usage{
torch_nan_to_num(self, nan = NULL, posinf = NULL, neginf = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{nan}{A numeric value. Optional.}

\item{posinf}{A numeric value. Optional.}

\item{neginf}{A numeric value. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nan_to_num}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
