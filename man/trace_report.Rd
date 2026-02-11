% tinyrox says don't edit this manually, but it can't stop you!
\name{trace_report}
\alias{trace_report}
\title{Try Tracing All Discovered Modules}
\usage{
trace_report(
  module,
  example_inputs = NULL,
  max_depth = 10L,
  check_correctness = TRUE,
  atol = 1e-04,
  verbose = TRUE
)
}
\arguments{
\item{module}{An nn_module instance (or a module_tree from discover_modules())}

\item{example_inputs}{Named list of example input tensors for the root module.
For sub-modules, inputs are inferred from forward() signatures using
tensor shapes derived from the root inputs.}

\item{max_depth}{Maximum depth to trace (default 10)}

\item{check_correctness}{If TRUE, runs both eager and traced and compares
outputs (default TRUE)}

\item{atol}{Absolute tolerance for correctness check (default 1e-4)}

\item{verbose}{Print progress (default TRUE)}
}
\value{
A data.frame with columns: path, class, n_params, graph_breaks,
  traceable, correct, error
}
\description{
Takes a module tree from discover_modules() and attempts trace_module()
on each sub-module. Reports graph breaks, correctness, and which modules
are candidates for optimization.
}
\examples{
\donttest{
m <- nn_sequential(nn_linear(10, 5), nn_relu())
trace_report(m, list(input = torch_randn(c(1, 10))))
}
}
