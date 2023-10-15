# Python, using Anaconda environment
# Week 3, Day 14
import helper_func.day14func as hfun



# Dungeon Crawler again

maze1 = hfun.Maze(7,7)
print(maze1)

maze1.init_adventurer_location()
print(maze1.Walker)
print(maze1.Walker.where_am_i())

maze1.Walker.go_up()
print(maze1.Walker.where_am_i())

maze1.Walker.play_dice()

"""x = (1, 3)
y = (0, 0)
print(id(x) == id(y))
y = (1, 3)
print(id(x) == id(y))"""

maze1.put_a_snake()
maze1.put_a_snake()
print(maze1.snakes)
for ix in maze1.snakes:
    print(ix)


