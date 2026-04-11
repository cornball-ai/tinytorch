% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_utils_clip_grad_norm_}
\alias{nn_utils_clip_grad_norm_}
\title{Clip gradient norm}
\usage{
nn_utils_clip_grad_norm_(parameters, max_norm, norm_type = 2)
}
\arguments{
\item{parameters}{List of tensors or an nn_module.}

\item{max_norm}{Maximum norm value.}

\item{norm_type}{Type of norm. Default 2.}
}
\value{
Total norm (numeric).
}
\description{
Clip gradient norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
