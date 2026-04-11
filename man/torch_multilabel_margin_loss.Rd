% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_multilabel_margin_loss}
\alias{torch_multilabel_margin_loss}
\title{Multilabel margin loss}
\usage{
torch_multilabel_margin_loss(self, target, reduction = 1L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::multilabel_margin_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
