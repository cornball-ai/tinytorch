#' Graph Segmentation
#'
#' Splits an expression at graph break points into segments that can
#' each be compiled independently, with R code executed between them.

#' Segment an Expression at Graph Breaks
#'
#' Takes a block expression and splits it into segments:
#' - "graph" segments contain consecutive graph-safe statements
#' - "r_code" segments contain graph-breaking statements
#'
#' @param expr An R expression (typically a block `{...}`)
#' @return A list of segments, each with `type` ("graph" or "r_code")
#'   and `statements` (list of expressions)
#' @examples
#' segs <- segment_expr(quote({ y <- x$relu(); print(y) }))
#' length(segs)
#' @export
segment_expr <- function(expr) {
  walked <- walk_expr(expr)

  # Handle non-block expressions

  if (walked$type != "block") {
    # Single expression - either all graph or all R
    if (is_graph_safe(walked)) {
      return(list(list(type = "graph", statements = list(expr))))
    } else {
      return(list(list(type = "r_code", statements = list(expr))))
    }
  }

  # Process block statements
  segments <- list()
  current_segment <- list(type = NULL, statements = list())

  for (i in seq_along(walked$statements)) {
    stmt_node <- walked$statements[[i]]
    # Get the original expression for this statement
    stmt_expr <- expr[[i + 1L]]  # +1 because expr[[1]] is `{`

    stmt_safe <- is_graph_safe(stmt_node)
    stmt_type <- if (stmt_safe) "graph" else "r_code"

    # Start new segment or continue current?
    if (is.null(current_segment$type)) {
      # First statement
      current_segment$type <- stmt_type
      current_segment$statements <- list(stmt_expr)
    } else if (current_segment$type == stmt_type) {
      # Same type - append to current segment
      current_segment$statements <- c(current_segment$statements, list(stmt_expr))
    } else {
      # Type changed - save current, start new
      segments <- c(segments, list(current_segment))
      current_segment <- list(type = stmt_type, statements = list(stmt_expr))
    }
  }

  # Don't forget the last segment
  if (length(current_segment$statements) > 0) {
    segments <- c(segments, list(current_segment))
  }

  segments
}


#' Analyze Expression Segments
#'
#' Provides a summary of how an expression would be segmented.
#'
#' @param expr An R expression
#' @return A data.frame with segment info
#' @examples
#' analyze_segments(quote({ y <- x$relu(); print(y) }))
#' @export
analyze_segments <- function(expr) {
  segments <- segment_expr(expr)

  data.frame(
    segment = seq_along(segments),
    type = vapply(segments, `[[`, character(1), "type"),
    n_statements = vapply(segments, function(s) length(s$statements), integer(1)),
    stringsAsFactors = FALSE
  )
}


#' Execute Segmented Expression
#'
#' Executes an expression by running graph segments through torch
#' and R code segments through normal evaluation. This is the
#' graph-break-aware execution engine.
#'
#' @param expr An R expression
#' @param env Environment containing tensors and other variables
#' @param compile Logical, whether to use jit_trace for graph segments.
#'   Superseded by execute_compiled() which uses the IR pipeline.
#' @return The result of the last statement
#' @noRd
execute_segmented <- function(expr, env, compile = FALSE) {
  segments <- segment_expr(expr)

  result <- NULL

  for (seg in segments) {
    # Both paths eval directly — compilation is handled by
    # execute_compiled() via the IR pipeline instead
    for (stmt in seg$statements) {
      result <- eval(stmt, envir = env)
    }
  }

  result
}


#' Print Segment Summary
#'
#' Pretty-prints how an expression would be segmented.
#'
#' @param expr An R expression
#' @return Invisibly returns the segments
#' @examples
#' print_segments(quote({ y <- x$relu(); print(y) }))
#' @export
print_segments <- function(expr) {
  segments <- segment_expr(expr)

  message(sprintf("Expression segments: %d", length(segments)))
  message(paste(rep("-", 40), collapse = ""))

  for (i in seq_along(segments)) {
    seg <- segments[[i]]
    type_label <- if (seg$type == "graph") "[GRAPH]" else "[R CODE]"

    message(sprintf("\nSegment %d %s (%d statement%s):",
                    i, type_label,
                    length(seg$statements),
                    if (length(seg$statements) == 1) "" else "s"))

    for (stmt in seg$statements) {
      lines <- deparse(stmt)
      for (line in lines) {
        message(sprintf("  %s", line))
      }
    }
  }

  invisible(segments)
}
