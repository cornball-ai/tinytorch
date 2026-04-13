% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_smooth_l1_loss}
\alias{nnf_smooth_l1_loss}
\title{Nnf smooth l1 loss}
\usage{
nnf_smooth_l1_loss(self, target, reduction, beta)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Reduction string: `"none"`, `"mean"`, or `"sum"`.}

\item{beta}{Numeric scalar multiplier.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf smooth l1 loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
