% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_utils_clip_grad_value_}
\alias{nn_utils_clip_grad_value_}
\title{Clip gradient values}
\usage{
nn_utils_clip_grad_value_(parameters, clip_value)
}
\arguments{
\item{parameters}{List of tensors or an nn_module.}

\item{clip_value}{Maximum absolute value.}
}
\description{
Clip gradient values
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
