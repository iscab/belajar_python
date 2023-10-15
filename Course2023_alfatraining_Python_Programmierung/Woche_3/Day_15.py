# Python, using Anaconda environment
# Week 3, Day 15
import helper_func.day15func as hfun

# Dungeon Crawler again
maze1 = hfun.Maze(7,6)
print(maze1)

maze1.init_adventurer_location()
print(maze1.Walker)
print(maze1.Walker.where_am_i())

maze1.Walker.go_up()
print(maze1.Walker.where_am_i())
print("\n")

maze1.Walker.play_dice()
print("\n")

# maze1.put_a_snake()
"""maze1.put_a_snake()
print(maze1.snakes)
for ix in maze1.snakes:
    print(ix)
print("\n")"""

# maze1.put_some_snakes(3)
print(maze1.snakes, type(maze1.snakes))
for ix in maze1.snakes:
    print(ix, type(ix))
print(maze1.Walker, type(maze1.Walker))
print("Adventurer:  ", maze1.Walker.where_am_i())
print("\n")





