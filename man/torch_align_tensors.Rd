% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_align_tensors}
\alias{torch_align_tensors}
\title{Align tensors}
\usage{
torch_align_tensors(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::align_tensors}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
