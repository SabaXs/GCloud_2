# Function to generate Fibonacci series
def generate_fibonacci(n):
    fibonacci_series = []
    a, b = 0, 1  # Initialize the first two Fibonacci numbers

    for _ in range(n):
        fibonacci_series.append(a)
        a, b = b, a + b  # Update Fibonacci numbers

    return fibonacci_series

# Number of terms in the Fibonacci series
num_terms = 5  # You can change this to the number of terms you want

# Generate and print the Fibonacci series
fib_series = generate_fibonacci(num_terms)
print("Fibonacci Series:")
for term in fib_series:
    print(term)
