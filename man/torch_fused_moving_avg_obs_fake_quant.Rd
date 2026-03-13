% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fused_moving_avg_obs_fake_quant}
\alias{torch_fused_moving_avg_obs_fake_quant}
\title{Fused moving avg obs fake quant}
\usage{
torch_fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on,
                                      running_min, running_max, scale,
                                      zero_point, averaging_const, quant_min,
                                      quant_max, ch_axis,
                                      per_row_fake_quant = FALSE,
                                      symmetric_quant = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{observer_on}{A torch_tensor.}

\item{fake_quant_on}{A torch_tensor.}

\item{running_min}{A torch_tensor.}

\item{running_max}{A torch_tensor.}

\item{scale}{A torch_tensor.}

\item{zero_point}{A torch_tensor.}

\item{averaging_const}{A numeric value.}

\item{quant_min}{An integer.}

\item{quant_max}{An integer.}

\item{ch_axis}{An integer.}

\item{per_row_fake_quant}{A logical value.}

\item{symmetric_quant}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fused_moving_avg_obs_fake_quant}.
}
