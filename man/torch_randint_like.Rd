% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randint_like}
\alias{torch_randint_like}
\title{Randint like}
\usage{
torch_randint_like(input, high, dtype = NULL, device = NULL,
                   memory_format = NULL)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{high}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}

\item{memory_format}{A memory format. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::randint_like}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
