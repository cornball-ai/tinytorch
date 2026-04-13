% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_hinge_embedding_loss}
\alias{nnf_hinge_embedding_loss}
\title{Nnf hinge embedding loss}
\usage{
nnf_hinge_embedding_loss(self, target, margin, reduction)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{margin}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf hinge embedding loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
