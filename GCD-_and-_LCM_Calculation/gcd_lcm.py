def gcd(a, b):
    """Compute the Greatest Common Divisor (GCD) of two numbers."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """"Compute the Least Common Multiple (LCM) of two numbers.""""
    return abs(a * b) // gcd(a, b)

# Main program
if __name__ == "__main__":
    print("GCD and LCM Calculator")
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))

    print(f"GCD of {a} and {b} is: {gcd(a, b)}")
    print(f"LCM of {a} and {b} is: {lcm(a, b)}")
