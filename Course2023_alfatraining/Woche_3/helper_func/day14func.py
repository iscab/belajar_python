# Python, using Anaconda environment
# Week 3, Day 14

import random
import datetime

# inheritance / Vererbung
class MeansOfTransportation:
    """ Main classfor means of transportation"""
    def __init__(self, speed: int, capacity: int, num_wheels: int, color: str):
        self.speed = speed
        self.capacity = capacity
        self.num_wheels = num_wheels
        self.color = color.upper()

    def __str__(self):
        return "Transport"

    def look(self):
        print(f"The vehicle has {self.num_wheels} wheels and has the color {self.color}")


class MotorCycle(MeansOfTransportation):
    """ Sub-class motorcycle of means of transportation """
    def __init__(self, speed, capacity, num_wheels, color):
        # Getting information from main class
        super().__init__(speed, capacity, num_wheels, color)

    def __str__(self):
        return "Motorcycle"

    def look(self):
        super().look()
        print("And is a motorcycle")


class Car(MeansOfTransportation):
    """ Sub-class car of means of transportation """
    def __init__(self, speed, capacity, num_wheels, color):
        # Getting information from main class
        super().__init__(speed, capacity, num_wheels, color)

    def __str__(self):
        return "Car"

    def look(self):
        super().look()
        print("And is a car")


class Bicycle(MeansOfTransportation):
    """ Sub-class bycicle of means of transportation """
    def __init__(self, speed, capacity, num_wheels, color):
        # Getting information from main class
        super().__init__(speed, capacity, num_wheels, color)

    def __str__(self):
        return "Bicycle"

    def look(self):
        super().look()
        print("And is a bicycle")


class Warrior:
    name =  "Warrior"
    health = 20
    attack = "Charge"


class Mage:
    name = "Mage"
    health = -20
    attack = "Fireball"


"""class AdventurerD(Warrior, Mage):
    def __init__(self, name: str, fightclass: str, bag: list):
        self.name = name
        if fightclass == "Warrior":
            self.fightclass = warrior.name
            self.maxHP =warrior.health"""





# Dungeon Crawler

class Adventurer:
    def __init__(self, name="Smith", pos_x=1, pos_y=1):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.num_of_step = 0

    def __str__(self):
        return f"My name is {self.name}, an adventurer"

    def where_am_i(self):
        return (self.pos_x, self.pos_y)

    def change_location(self, pos_x=0, pos_y=0):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def go_left(self):
        self.pos_x -= 1
        self.num_of_step -= 1

    def go_right(self):
        self.pos_x += 1
        self.num_of_step -= 1

    def go_up(self):
        self.pos_y += 1
        self.num_of_step -= 1

    def go_down(self):
        self.pos_y -= 1
        self.num_of_step -= 1

    def play_dice(self):
        self.num_of_step = random.randint(1,6)
        print(self.num_of_step)



class Snake:
    def __init__(self, head_pos_x=1, head_pos_y=2, tail_pos_x=2, tail_pos_y=1):
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
        # self.snake = Snake()


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
        # print(self.about_maze)

    def put_random_thing_in_maze(self):
        x = random.randint(1,self.size_x)
        y = random.randint(1, self.size_y)
        return (x, y)

    def init_adventurer_location(self):
        x, y = self.put_random_thing_in_maze()
        self.Walker.change_location(x, y)

    def put_a_snake(self):
        a_snake = Snake()
        x_head, y_head = self.put_random_thing_in_maze()
        x_tail, y_tail = self.put_random_thing_in_maze()

        a_snake.put_snake_head(x_head, y_head)
        a_snake.put_snake_tail(x_tail, y_tail)

        self.snakes.append(a_snake)



