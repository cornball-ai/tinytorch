% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_column_stack}
\alias{torch_column_stack}
\title{Column stack}
\usage{
torch_column_stack(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::column_stack}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
