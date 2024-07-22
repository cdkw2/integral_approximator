def fx(x):
    return x ** 2

def main():
    a = float(input("Enter the lower limit a: "))
    b = float(input("Enter the upper limit b: "))
    n = float(input("Enter the value of n: "))

    h = (b - a) / n
    p = a
    sum = 0
    
    while p < b - h:
        p += h
        jl = fx(p)
        sum += jl

    m = (h / 2) * (fx(a) + 2 * sum + fx(b))
    print("Result of integral approximation with the trapezoidal method: {:.7f}".format(m))

if __name__ == "__main__":
    main()
