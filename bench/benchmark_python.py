import time, json, torch

def timeit(fn, n=10000):
    for _ in range(100): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e6

x = torch.randn(10, 10)
big = torch.randn(1000, 1000)

results = {
    "namespace_add":  timeit(lambda: torch.add(x, x)),
    "method_add":     timeit(lambda: x.add(x)),
    "chained_matmul": timeit(lambda: torch.matmul(torch.matmul(x, x), x)),
    "method_chain":   timeit(lambda: x.add(x).mul(x)),
    "creation":       timeit(lambda: torch.randn(10, 10)),
    "large_matmul":   timeit(lambda: torch.matmul(big, big), n=200),
}

print(json.dumps(results))
