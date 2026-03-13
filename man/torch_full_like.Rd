% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_full_like}
\alias{torch_full_like}
\title{Full like}
\usage{
torch_full_like(self, fill_value, dtype = NULL, device = NULL,
                memory_format = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{fill_value}{A numeric scalar.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}

\item{memory_format}{A memory format. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::full_like}.
}
