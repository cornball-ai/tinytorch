% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_cross_entropy}
\alias{nnf_cross_entropy}
\title{Nnf cross entropy}
\usage{
nnf_cross_entropy(input, target, weight = NULL, reduction = 1L,
                  ignore_index = -100, label_smoothing = 0)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{ignore_index}{Parameter passed to the underlying ATen operator.}

\item{label_smoothing}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf cross entropy
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
