import multiprocessing
import time

def compute_square(n):
    total = 0
    for i in range(1000000):
        total += n * n
    return total

def main():
    numbers = [1, 2, 3, 4, 5]
    start_time = time.time()

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(compute_square, numbers)

    end_time = time.time()

    print("Results:", results)
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
