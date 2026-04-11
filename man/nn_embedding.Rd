% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_embedding}
\alias{nn_embedding}
\title{Embedding lookup module}
\usage{
nn_embedding(num_embeddings, embedding_dim, padding_idx = NULL)
}
\arguments{
\item{num_embeddings}{Integer, size of the dictionary.}

\item{embedding_dim}{Integer, size of each embedding vector.}

\item{padding_idx}{Optional integer. If given, pads output at this index.}
}
\value{
An nn_module instance.
}
\description{
Embedding lookup module
}
\examples{
\donttest{
if (torch_is_installed()) {
# an Embedding module containing 10 tensors of size 3
embedding <- nn_embedding(10, 3)
# a batch of 2 samples of 4 indices each
input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
embedding(input)
# example with padding_idx
embedding <- nn_embedding(10, 3, padding_idx = 1)
input <- torch_tensor(matrix(c(1, 3, 1, 6), nrow = 1), dtype = torch_long())
embedding(input)
}
}
}
