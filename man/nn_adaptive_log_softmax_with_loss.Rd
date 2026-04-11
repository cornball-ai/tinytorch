% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_adaptive_log_softmax_with_loss}
\alias{nn_adaptive_log_softmax_with_loss}
\title{Adaptive log-softmax with loss}
\usage{
nn_adaptive_log_softmax_with_loss(in_features, n_classes, cutoffs,
                                  div_value = 4, head_bias = FALSE)
}
\arguments{
\item{in_features}{Input feature size.}

\item{n_classes}{Number of classes.}

\item{cutoffs}{Cutoff values for adaptive softmax.}

\item{div_value}{Division value. Default 4.}

\item{head_bias}{Use bias in head. Default FALSE.}
}
\value{
An nn_module.
}
\description{
Adaptive log-softmax with loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
