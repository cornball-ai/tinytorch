% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nested_to_padded_tensor}
\alias{torch_nested_to_padded_tensor}
\title{Nested to padded tensor}
\usage{
torch_nested_to_padded_tensor(self, padding, output_size = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{padding}{A numeric value.}

\item{output_size}{An integer vector. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nested_to_padded_tensor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
