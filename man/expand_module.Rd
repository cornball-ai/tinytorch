% tinyrox says don't edit this manually, but it can't stop you!
\name{expand_module}
\alias{expand_module}
\title{Expand a Module's forward() Body}
\usage{
expand_module(
  module,
  arg_exprs = list(),
  prefix = "",
  depth = 0L,
  max_depth = 10L
)
}
\arguments{
\item{module}{An nn_module callable or R6 instance}

\item{arg_exprs}{Named list of argument expressions}

\item{prefix}{Dot-separated name prefix for parameters}

\item{depth}{Current recursion depth (to prevent infinite loops)}

\item{max_depth}{Maximum recursion depth}
}
\value{
A list with: statements (list of rewritten exprs),
  params (named list of parameter tensors),
  graph_breaks (list of break descriptions)
}
\description{
Recursively rewrites the AST of a module's forward method:
- self$param -> unique parameter symbol
- self$submodule(args) -> recursive expansion
- nnf_foo(args) -> torch_* call
- static if/else -> resolved branch
- nn_sequential for loop -> unrolled
}
