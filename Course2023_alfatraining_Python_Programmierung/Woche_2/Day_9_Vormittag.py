# Python, using Anaconda environment
# Week 2, Day 9


# recursion

def fractional(n):
    if n == 1:
        print("n==1")
        print(n)
        return 1
    if n > 1:
        print("n>1")
        print(n)
        return n * fractional(n - 1)


test_out = fractional(3)
print("output = ", test_out)
print("\n")


def sum_until(n):
    n = int(n)
    mySum = 0
    for ix in range(0, n+1):
        mySum += ix
    return mySum

def sum_till(n):
    n = int(n)
    if n<1:
        return 0
    else:
        return n + sum_till(n - 1)


def sum_plus_minus(n):
    n = int(n)
    print(n)
    if n == 0 or n == 1:
        return n
    if n > 1:
        return n + sum_plus_minus(n - 1)
    if n < 0:
        return n + sum_plus_minus(n + 1)


x = 5
test_out = sum_until(x)
print("looping output = ", test_out)
test_out = sum_till(x)
print("recursive output = ", test_out)
print("\n")
test_out = sum_plus_minus(x)
print("recursive output = ", test_out)
test_out = sum_plus_minus(-x)
print("recursive output = ", test_out)
print("\n")

# watch 6 minutes video about sorting algorithms:  https://www.youtube.com/watch?v=kPRA0W1kECg
# read this:  https://en.wikipedia.org/wiki/Merge_sort










