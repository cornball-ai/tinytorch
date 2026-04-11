% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_multi_dot}
\alias{torch_linalg_multi_dot}
\alias{linalg_multi_dot}
\title{Linalg multi dot}
\usage{
torch_linalg_multi_dot(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_multi_dot}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
