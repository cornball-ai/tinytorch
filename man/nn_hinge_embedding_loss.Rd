% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_hinge_embedding_loss}
\alias{nn_hinge_embedding_loss}
\title{Nn hinge embedding loss}
\usage{
nn_hinge_embedding_loss(margin = 1, reduction = 1L)
}
\arguments{
\item{margin}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn hinge embedding loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
