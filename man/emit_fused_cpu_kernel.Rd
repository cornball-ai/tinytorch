% tinyrox says don't edit this manually, but it can't stop you!
\name{emit_fused_cpu_kernel}
\alias{emit_fused_cpu_kernel}
\title{Emit Fused CPU Kernel C++ Code from IR}
\usage{
emit_fused_cpu_kernel(graph, group_id, func_name = NULL)
}
\arguments{
\item{graph}{An ir_graph with fusion annotations}

\item{group_id}{Integer fusion group ID}

\item{func_name}{Optional function name (auto-generated if NULL)}
}
\value{
List with code, func_name, n_inputs, external_input_ids,
  output_id, group_node_ids, dtype. NULL if group has unsupported ops.
}
\description{
Takes an IR graph and a fusion group ID. Extracts the group's nodes,
determines external inputs, and generates a single C++ function with
SIMD-vectorized loop + scalar tail using at::vec API.
}
