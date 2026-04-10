% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scaled_mm}
\alias{torch_scaled_mm}
\title{Scaled matrix multiplication (FP8)}
\usage{
torch_scaled_mm(self, mat2, scale_a, scale_b, bias = NULL, scale_result = NULL,
                out_dtype = NULL, use_fast_accum = FALSE)
}
\arguments{
\item{self}{First matrix (float8 tensor).}

\item{mat2}{Second matrix (float8 tensor).}

\item{scale_a}{Scale factor for \code{self} (scalar tensor).}

\item{scale_b}{Scale factor for \code{mat2} (scalar tensor).}

\item{bias}{Optional bias tensor.}

\item{scale_result}{Optional scale factor for the output.}

\item{out_dtype}{Output dtype (e.g., \code{torch_bfloat16}). Default NULL uses input dtype.}

\item{use_fast_accum}{Use fast accumulation (less precise, faster). Default FALSE.}
}
\value{
A torch_tensor.
}
\description{
Performs matrix multiplication with separate per-tensor scaling factors,
designed for FP8 (float8) training and inference. Inputs are typically
float8_e4m3fn or float8_e5m2 tensors; output is cast to \code{out_dtype}.
}
