% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_upsample_nearest1d}
\alias{torch_upsample_nearest1d}
\title{Upsample nearest1d}
\usage{
torch_upsample_nearest1d(input, output_size, scale_factors)
}
\arguments{
\item{input}{A torch_tensor.}

\item{output_size}{An integer vector.}

\item{scale_factors}{A numeric vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::upsample_nearest1d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
