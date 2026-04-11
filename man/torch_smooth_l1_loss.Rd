% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_smooth_l1_loss}
\alias{torch_smooth_l1_loss}
\title{Smooth l1 loss}
\usage{
torch_smooth_l1_loss(self, target, reduction = 1L, beta = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{reduction}{An integer.}

\item{beta}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::smooth_l1_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
