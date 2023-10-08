# Python, using Anaconda environment
# Week 4, Day 16
from dataclasses import dataclass

import random
import datetime


# read this:  https://docs.python.org/3/library/dataclasses.html
# read this:  https://www.pythontutorial.net/python-oop/python-dataclass/
# read this:  https://www.dataquest.io/blog/how-to-use-python-data-classes/


@dataclass()
class Teilnehmer:
    name: str
    age: int
    occupation: str

    def my_age(self):
        print(f"{self.name} is {self.age} years old")

    def my_job(self):
        print(f"{self.name} is a {self.occupation}")


class Member:
    def __init__(self, name, age, occupation):
        self.name = name
        self.age = age
        self.occupation = occupation

    def my_age(self):
        print(f"{self.name} is {self.age} years old")

    def my_job(self):
        print(f"{self.name} is a {self.occupation}")


class durchzahlen:
    # creating the attributes -> an iterable variable
    def __init__(self, iterableVar):
        self.iterableVar = iterableVar
        self.iter_obj = iter(iterableVar)

    # definition of iteration (beginning of iteration)
    def __iter__(self):
        return self

    # definition on what should be the next element to iterate
    def __next__(self):
        while True:
            try:
                next_obj = next(self.iter_obj)
                return next_obj
            except:
                self.iter_obj = iter(self.iterableVar)



# Dungeon Crawler

class Adventurer:
    def __init__(self, name: str = "Smith", pos_x: int = 1, pos_y: int = 1):
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
        if self.does_walker_have_remaining_step():
            self.pos_x -= 1
            self.num_of_step -= 1
            print(f"Walker go left a step, and now at ({self.pos_x},{self.pos_y}). \n")

    def go_right(self):
        if self.does_walker_have_remaining_step():
            self.pos_x += 1
            self.num_of_step -= 1
            print(f"Walker go right a step, and now at ({self.pos_x},{self.pos_y}). \n")

    def go_up(self):
        if self.does_walker_have_remaining_step():
            self.pos_y += 1
            self.num_of_step -= 1
            print(f"Walker go up a step, and now at ({self.pos_x},{self.pos_y}). \n")

    def go_down(self):
        if self.does_walker_have_remaining_step():
            self.pos_y -= 1
            self.num_of_step -= 1
            print(f"Walker go down a step, and now at ({self.pos_x},{self.pos_y}). \n")

    def play_dice(self):
        self.num_of_step = random.randint(1,6)
        print(self.num_of_step)

    def does_walker_have_remaining_step(self):
        print(f"Walker is now at ({self.pos_x},{self.pos_y})")
        if self.num_of_step < 1:
            print("No remaining steps. Please roll the dice! ")
            return False
        else:
            print(f"remaining steps:  {self.num_of_step}")
            return True



class Snake:
    def __init__(self, head_pos_x: int = 1, head_pos_y: int = 2, tail_pos_x: int = 2, tail_pos_y: int = 1):
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

    def put_snake_head(self, head_pos_x: int = 0, head_pos_y: int = 0):
        self.head_pos_x = head_pos_x
        self.head_pos_y = head_pos_y
        self.about = f"I am a snake, spreading my head from ({self.head_pos_x},{self.head_pos_y}) until my tail at ({self.tail_pos_x},{self.tail_pos_y})"

    def put_snake_tail(self, tail_pos_x: int = 0, tail_pos_y: int =0):
        self.tail_pos_x = tail_pos_x
        self.tail_pos_y = tail_pos_y
        self.about = f"I am a snake, spreading my head from ({self.head_pos_x},{self.head_pos_y}) until my tail at ({self.tail_pos_x},{self.tail_pos_y})"


class Maze:
    def __init__(self, size_x: int = 8, size_y: int = 8):
        # maze dimension
        self.size_x = 0
        self.size_y = 0
        self.about_maze = f"Maze with the size {self.size_x} X {self.size_y} "

        # snakes
        self.snakes = []  # list of snakes
        self.num_of_snakes = 3
        self.set_area(size_x, size_y)
        # self.snake = Snake()

        # adventurer
        self.Walker = Adventurer()
        self.init_adventurer_location()

    def __str__(self):
        return self.about_maze

    def set_area(self, size_x: int = 8, size_y: int = 8):
        self.size_x = max(size_x, 5)
        self.size_y = max(size_y, 5)
        self.about_maze = f"Maze with the size {self.size_x} X {self.size_y} "
        self.num_of_snakes = min(self.size_y, self.size_x) - 1
        self.put_some_snakes(self.num_of_snakes)
        # print(self.about_maze)

    def put_random_thing_in_maze(self):
        x = random.randint(1, self.size_x)
        y = random.randint(1, self.size_y)
        return (x, y)

    def init_adventurer_location(self):
        x, y = self.put_random_thing_in_maze()
        self.Walker.change_location(x, y)

    def put_a_snake(self):
        add_another_snake = True
        a_snake = Snake()
        x_head, y_head = self.put_random_thing_in_maze()
        x_tail, y_tail = self.put_random_thing_in_maze()
        # print("head: ", x_head, y_head)
        # print("tail: ",x_tail, y_tail)

        # check other snakes
        if len(self.snakes) > 0:
            for idx, other_snake in enumerate(self.snakes):
                # print(idx, other_snake, type(other_snake))
                # print(idx)
                add_another_snake = add_another_snake and (x_head, y_head) != (x_tail, y_tail)
                x, y = other_snake.where_is_head()
                add_another_snake = add_another_snake and (x, y) != (x_head, y_head)
                x, y = other_snake.where_is_tail()
                add_another_snake = add_another_snake and (x, y) != (x_head, y_head)
                if not add_another_snake:
                    break
        else:
            add_another_snake = True

        if add_another_snake:
            a_snake.put_snake_head(x_head, y_head)
            a_snake.put_snake_tail(x_tail, y_tail)
            self.snakes.append(a_snake)
        else:
            self.put_a_snake()

    def put_some_snakes(self, num_of_snakes: int):
        for snake_idx in range(num_of_snakes):
            # print(snake_idx)
            self.put_a_snake()

    def let_adventure_go_up(self):
        if self.Walker.pos_y != self.size_y:
            self.Walker.go_up()
        else:
            print("There is a wall, walker can't go up. ")

    def let_adventure_go_down(self):
        if self.Walker.pos_y != 1:
            self.Walker.go_down()
        else:
            print("There is a wall, walker can't go down. ")

    def let_adventure_go_left(self):
        if self.Walker.pos_x != 1:
            self.Walker.go_left()
        else:
            print("There is a wall, walker can't go left. ")

    def let_adventure_go_right(self):
        if self.Walker.pos_x != self.size_x:
            self.Walker.go_right()
        else:
            print("There is a wall, walker can't go right. ")

    def does_adventure_meet_snake(self):
        not_meet_snake = True
        walker_x, walker_y = self.Walker.where_am_i()
        # print(walker_x, walker_y)

        for snake in self.snakes:
            snake_head_x, snake_head_y = snake.where_is_head()
            # print(snake_head_x, snake_head_y)
            not_meet_snake = not_meet_snake and (walker_x, walker_y) != (snake_head_x, snake_head_y)
            # print(not_meet_snake)

        return not not_meet_snake





