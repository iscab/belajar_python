# Python, using Anaconda environment
# Week 2, Day 6
import time

# Part 1
print("Part 1: Loops and Statements  \n")

# Exercise 1
print("Exercise 1: for-loop  \n")
for ix in range(1,30):
    # print even numbers between 1 and 30
    if ix % 2 == 0:
        print(ix)
# TODO: try option random
print("\n")

# Exercise 2
print("Exercise 2: while-loop  \n")
mySum = 0
last_added_number = 0
ix = 1
while ix <= 100:
    last_added_number = ix
    if mySum + last_added_number < 1000:
        # calculate sum as long as it is not larger than 1000
        mySum += last_added_number  # calculate sum
    else:
        last_added_number -= 1
        break
    ix += 1
print("The sum is ", mySum)
print("the last added number is ", last_added_number)
print("\n")

# Exercise 3
print("Exercise 3: Fibonacci values  \n")
fibonacci_series = [0, 1]
f0_value = fibonacci_series[0]
f1_value = fibonacci_series[1]
f_next_value = 0
max_limit = 100
while f_next_value < max_limit:
    f_next_value = f1_value + f0_value
    f0_value = f1_value
    f1_value = f_next_value
    if f_next_value >= max_limit:
        break
    else:
        fibonacci_series.append(f_next_value)
print("Fibonacci Series:  ", fibonacci_series)
print("\n")

# Exercise 4
print("Exercise 4: prime numbers  \n")
prime_list = []
max_limit = 10
for ix in range(2, max_limit):
    print(ix)
    number_div = 1
    denum = ix
    min_check = 1
    while denum > min_check:
        if ix % denum == 0:
            print(ix, denum, number_div)
        denum -= 1
        number_div -= 1
        if number_div <= -1:
            break
# TODO: just do it
print("\n")

# exit()  # debug


# Exercise 5
print("Exercise 5: password checking  ")
myPassWord = "Bajingan42"
myInput = "Bajingan42"  # right password
# myInput = "AbCd1234"
if myInput == myPassWord and isinstance(myInput, str):
    print("The password is correct")
else:
    print("The password is not correct")
print("\n")

# Exercise 6
print("Exercise 6: ")
start_time = time.time()
# print(start_time)
laufzeit = []
for ix in range(1,30):
    # even numbers between 1 and 30
    time.sleep(0.01)  # seconds
    if ix % 2 == 0:
        idx_0 = ix
        idx_1 = time.time() - start_time
        # print(idx_1)
        laufzeit.append([idx_0, idx_1])
print(laufzeit)
# TODO: graphical plot
print("\n")

# Part 2
print("Part 2: Functions  \n")

# Exercise 7
print("Exercise 7:  sum of numbers")


def sum_till(x):
    """
    the sum from 0 to X
    :param x: (int) number at the end
    :return: mySum as the sum
    """
    mySum = 0
    for ix in range(0, x):
        mySum += ix

    return mySum


x_input = 10
print(f"The sum of 0 to {x_input} is ", sum_till(x_input))
print("\n")

# Exercise 8
print("Exercise 8: area of a square")


def area_of_square(edge_length):
    """
    calculate the area of a square with the input of edge length
    :param edge_length: (int/float) the edge length of the square
    :return: area (int/float)
    """
    area = edge_length**2
    return area


x_input = 15.3
my_area = area_of_square(x_input)
print("edge length = ", x_input, " will have a square area of ", my_area)
print("\n")


# Exercise 9
print("Exercise 9: string output with frame  \n")


def beautiful_output(myText, txtWidth):
    """
    write text in a frame
    :param myText: (str) the input text
    :param txtWidth: (int) the width of the desired output
    :return:
    """
    min_indent = 4
    # width calculation
    min_width = len(myText) + 2 * min_indent
    if txtWidth > min_width:
        width = txtWidth
    else:
        width = min_width
    my_indent = int(0.5*(width + 1 - len(myText)))

    # write text
    out_string = ""
    frame = ""
    content = ""
    for ix in range(0,my_indent):
        frame += "-"
        content += " "

    content += myText
    for ix in range(0,len(myText)):
        frame += "-"

    for ix in range(0,my_indent):
        frame += "-"
        content += " "

    out_string += "+" + frame + "+\n"
    out_string += "+" + content + "+\n"
    out_string += "+" + frame + "+\n"

    return out_string


# input_text = "Jojon pergi ke pasar. Pasar apa?"
input_text = "Juhu!"
output_text = beautiful_output(input_text, 20)
print(output_text)
