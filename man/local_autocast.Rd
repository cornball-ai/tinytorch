% tinyrox says don't edit this manually, but it can't stop you!
\name{local_autocast}
\alias{local_autocast}
\title{Local autocast}
\usage{
local_autocast(device_type = "cuda", dtype = torch_float16,
               .local_envir = parent.frame())
}
\arguments{
\item{device_type}{Parameter passed to the underlying ATen operator.}

\item{dtype}{A `torch_dtype`.}

\item{.local_envir}{Parameter passed to the underlying ATen operator.}
}
\description{
Local autocast
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
