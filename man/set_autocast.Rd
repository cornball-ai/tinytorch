% tinyrox says don't edit this manually, but it can't stop you!
\name{set_autocast}
\alias{set_autocast}
\title{Set autocast}
\usage{
set_autocast(enabled, device_type = "cuda", dtype = torch_float16)
}
\arguments{
\item{enabled}{Parameter passed to the underlying ATen operator.}

\item{device_type}{Parameter passed to the underlying ATen operator.}

\item{dtype}{A `torch_dtype`.}
}
\value{
No return value, called for side effects.
}
\description{
Set autocast
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
