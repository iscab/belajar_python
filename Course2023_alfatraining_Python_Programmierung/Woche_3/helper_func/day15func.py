# Python, using Anaconda environment
# Week 3, Day 15

import random
import datetime


# inheritance / Vererbung

class Human:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.about = f"I am {self.name}, {self.age} years old "

    def __str__(self):
        return self.about

    def mobility(self):
        return "I am walking"


class Baby(Human):
    def __init__(self, name, age):
        super().__init__(name, age)
        self.about += "and I am still a baby."

    def mobility(self):
        return "I am crawling"


class Toddler(Human):
    def __init__(self, name, age):
        super().__init__(name, age)
        self.about += "and I am still a toddler."

    def mobility(self):
        return "I am toddling"



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






