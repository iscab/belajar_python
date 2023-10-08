# Python, using Anaconda environment
# Week 3, Day 13
import helper_func.day13func as hfun

# Dungeon Crawler

# test adventurer
walker1 = hfun.Adventurer("Juan", 1,1)
print(walker1)
print(walker1.where_am_i())
walker1.change_location(2,3)
print(walker1.where_am_i())
print("\n")

# test snakes
snake1 = hfun.Snake(1,2,3,5)
print(snake1)
print(snake1.where_is_head())
print(snake1.where_is_tail())
snake1.put_snake_head(2, 3)
print(snake1)
print(snake1.where_is_head())
print(snake1.where_is_tail())
snake1.put_snake_tail(3,1)
print(snake1)
print(snake1.where_is_head())
print(snake1.where_is_tail())
print("\n")

snake0 = hfun.Snake()
print(snake0)
print(snake0.where_is_head())
print(snake0.where_is_tail())
print("\n")


# test maze
maze1 = hfun.Maze(7,7)
print(maze1)
print(maze1.num_of_snakes)
maze1.set_area(8,8)
print(maze1)
print(maze1.num_of_snakes)
maze1.set_area(3,3)
print(maze1)
print(maze1.num_of_snakes)
maze1.set_area(-3,3)
print(maze1)
print(maze1.num_of_snakes)
print("\n")

x, y = maze1.put_random_thing_in_maze()
print(x, y)
