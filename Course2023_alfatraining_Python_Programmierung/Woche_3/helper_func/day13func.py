# Python, using Anaconda environment
# Week 3, Day 13

import random

class Dog:
    def __init__(self, name="Blacky", breed="Mongrel", age=6, color="mostly black"):
        # self.name = name.upper()
        self.name = name.capitalize()
        self.breed = breed.lower()
        self.age = age + 1
        self.color = color

    def __str__(self):
        return f"{self.name}, {self.age}. A {self.breed} with the color {self.color}"

    def eat(self, food):
        if not type(food) == str:
            raise TypeError("Food has to be a string")
        return f"{self.name} is eating {food}, it's awesome! "

    def __add__(self, other):
        return Dog(self.name + " " + other.name, self.breed + "-" + other.breed, 1, self.color + " and " + other.color)


class Ninja:
    def __init__(self, name="Naruto", weapon="Shuriken", jutsu= "Kage Bunshin", level=5):
        self.name = name
        self.weapon = weapon
        self.jutsu = jutsu
        self.level = level
        self.about = f"{self.name}, ninja level {self.level}, expert of {self.weapon} and {self.jutsu}"
        self.pos_x = 0
        self.pos_y = 0

    def __str__(self):
        return self.about

    def my_position(self):
        return (self.pos_x, self.pos_y)

    def initial_position(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def go_left(self):
        self.pos_x -= 1

    def go_right(self):
        self.pos_x += 1

    def go_up(self):
        self.pos_y += 1

    def go_down(self):
        self.pos_y -= 1


# Dungeon Crawler

class Adventurer:
    def __init__(self, name="Smith", pos_x=0, pos_y=0):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y

    def __str__(self):
        return f"My name is {self.name}, an adventurer"

    def where_am_i(self):
        return (self.pos_x, self.pos_y)

    def change_location(self, pos_x=0, pos_y=0):
        self.pos_x = pos_x
        self.pos_y = pos_y


class Snake:
    def __init__(self, head_pos_x=0, head_pos_y=0, tail_pos_x=1, tail_pos_y=1):
        self.head_pos_x = head_pos_x
        self.head_pos_y = head_pos_y
        self.tail_pos_x = tail_pos_x
        self.tail_pos_y = tail_pos_y
        self.about = f"I am a snake, spreading my head from ({self.head_pos_x},{self.head_pos_y}) until my tail at ({self.tail_pos_x},{self.tail_pos_y})"

    def __str__(self):
        return self.about

    def where_is_head(self):
        return (self.head_pos_x, self.head_pos_y)

    def where_is_tail(self):
        return (self.tail_pos_x, self.tail_pos_y)

    def put_snake_head(self, head_pos_x=0, head_pos_y=0):
        self.head_pos_x = head_pos_x
        self.head_pos_y = head_pos_y
        self.about = f"I am a snake, spreading my head from ({self.head_pos_x},{self.head_pos_y}) until my tail at ({self.tail_pos_x},{self.tail_pos_y})"

    def put_snake_tail(self, tail_pos_x=0, tail_pos_y=0):
        self.tail_pos_x = tail_pos_x
        self.tail_pos_y = tail_pos_y
        self.about = f"I am a snake, spreading my head from ({self.head_pos_x},{self.head_pos_y}) until my tail at ({self.tail_pos_x},{self.tail_pos_y})"


class Maze:
    def __init__(self, size_x=8, size_y=8):
        # maze dimension
        self.size_x = 0
        self.size_y = 0
        self.about_maze = f"Maze with the size {self.size_x} X {self.size_y} "

        # snakes
        self.num_of_snakes = 3
        self.set_area(size_x, size_y)
        self.snakes = []  # list of snakes
        self.snake = Snake()


        # adventurer
        self.Walker = Adventurer()
        self.init_adventurer_location()

    def __str__(self):
        return self.about_maze

    def set_area(self, size_x=8, size_y=8):
        self.size_x = max(size_x, 5)
        self.size_y = max(size_y, 5)
        self.num_of_snakes = min(self.size_y, self.size_x) - 1
        self.about_maze = f"Maze with the size {self.size_x} X {self.size_y} "
        print(self.about_maze)

    def put_random_thing_in_maze(self):
        x = random.randint(0,self.size_x - 1)
        y = random.randint(0, self.size_y - 1)
        return (x, y)

    def init_adventurer_location(self):
        x, y = self.put_random_thing_in_maze()
        self.Walker.change_location(x, y)



