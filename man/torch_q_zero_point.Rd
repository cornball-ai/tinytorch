% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_q_zero_point}
\alias{torch_q_zero_point}
\title{Q zero point}
\usage{
torch_q_zero_point(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::q_zero_point}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
