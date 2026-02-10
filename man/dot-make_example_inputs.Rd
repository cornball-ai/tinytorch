% tinyrox says don't edit this manually, but it can't stop you!
\name{.make_example_inputs}
\alias{.make_example_inputs}
\title{Make Example Inputs for a Sub-module}
\usage{
.make_example_inputs(mod, arg_names, root_inputs = NULL)
}
\arguments{
\item{mod}{nn_module instance}

\item{arg_names}{Character vector of forward() argument names}

\item{root_inputs}{Optional named list of tensors (from user)}
}
\value{
Named list of tensors, or NULL if we can't figure it out
}
\description{
Tries to create reasonable tensor inputs for a module's forward() args.
Uses a small fixed shape (1, 16, dim) where dim is guessed from the
module's first parameter's shape.
}
\keyword{internal}
