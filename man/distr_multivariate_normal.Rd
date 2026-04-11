% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_multivariate_normal}
\alias{distr_multivariate_normal}
\title{Multivariate normal distribution}
\usage{
distr_multivariate_normal(loc, covariance_matrix = NULL,
                          precision_matrix = NULL, scale_tril = NULL)
}
\arguments{
\item{loc}{Mean vector (tensor).}

\item{covariance_matrix}{Covariance matrix (tensor). One of covariance_matrix,
precision_matrix, or scale_tril required.}

\item{precision_matrix}{Precision matrix (tensor).}

\item{scale_tril}{Lower-triangular scale matrix (tensor).}
}
\value{
A distribution object.
}
\description{
Multivariate normal distribution
}
