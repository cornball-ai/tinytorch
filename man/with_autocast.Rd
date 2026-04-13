% tinyrox says don't edit this manually, but it can't stop you!
\name{with_autocast}
\alias{with_autocast}
\title{No-op autocast context manager}
\usage{
with_autocast(device_type = "cpu", dtype = NULL, enabled = TRUE, code)
}
\arguments{
\item{device_type}{Ignored.}

\item{dtype}{Ignored.}

\item{enabled}{Ignored.}

\item{code}{Expression to evaluate.}
}
\value{
The result of evaluating `code`.
}
\description{
Since tinytorch is CPU-only, autocast is a no-op.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
