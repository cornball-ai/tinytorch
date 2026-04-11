% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_batch_norm2d}
\alias{nn_batch_norm2d}
\title{2D batch normalization module}
\usage{
nn_batch_norm2d(num_features, eps = 1e-05, momentum = 0.1, affine = TRUE,
                track_running_stats = TRUE)
}
\arguments{
\item{num_features}{Integer.}

\item{eps}{Double. Default 1e-5.}

\item{momentum}{Double. Default 0.1.}

\item{affine}{Logical. Default TRUE.}

\item{track_running_stats}{Logical. Default TRUE.}
}
\value{
An nn_module instance.
}
\description{
2D batch normalization module
}
\examples{
\donttest{
if (torch_is_installed()) {
# With Learnable Parameters
m <- nn_batch_norm2d(100)
# Without Learnable Parameters
m <- nn_batch_norm2d(100, affine = FALSE)
input <- torch_randn(20, 100, 35, 45)
output <- m(input)
}
}
}
