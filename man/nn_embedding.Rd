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
