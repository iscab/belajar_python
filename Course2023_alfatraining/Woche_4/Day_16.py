# Python, using Anaconda environment
# Week 4, Day 16
import helper_func.day16func as hfun


# Dungeon Crawler again
maze1 = hfun.Maze(7,7)
print(maze1)
print("\n")

print(maze1.snakes, type(maze1.snakes))
for ix in maze1.snakes:
    print(ix, type(ix))
print(maze1.Walker, type(maze1.Walker))
print("Adventurer:  ", maze1.Walker.where_am_i())
print("\n")


maze1.Walker.play_dice()

"""maze1.let_adventure_go_up()
print("Adventurer:  ", maze1.Walker.where_am_i())

maze1.let_adventure_go_up()
print("Adventurer:  ", maze1.Walker.where_am_i())"""

"""maze1.let_adventure_go_down()
# print("Adventurer:  ", maze1.Walker.where_am_i())

maze1.let_adventure_go_down()
# print("Adventurer:  ", maze1.Walker.where_am_i())"""

"""maze1.let_adventure_go_left()
maze1.let_adventure_go_left()
maze1.let_adventure_go_left()"""

"""maze1.let_adventure_go_right()
maze1.let_adventure_go_right()
maze1.let_adventure_go_right()"""

maze1.let_adventure_go_right()
maze1.does_adventure_meet_snake()
