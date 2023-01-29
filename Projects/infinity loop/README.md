# Infinity Loop

Infinity loop is a mobile app (also available here: https://poki.com/en/g/infinity-loop) where you rotate blocks until the shapes on them form a continuous loop. I recreated this in Python because I felt like it.

This was created using pygame. It was also one of my first Python projects, so I did not care about the UI, so it is not pretty.

A command-line interface should open up immediately as `infinity_loop_redux.py` is run, which allows three commands:
- `grid x y` - set game grid size to x by y (default is 10 x 10, with a minimum grid area of 8)
- `screen x y` - set the screen size to x by y pixels (default is 800 x 800, minimum is 50 x 50)
- `play` - start the game

The game is guaranteed to randomly generate a solveable game of any size, as well as center and proportion the sizes of blocks to fit to the screen. When playing, simply click the grid to flip the blocks. There are no animations for this sadly.

<p align="center">
<img src="small.gif" width="450" height="450"/>
</p>

The wonky black at the bottom-right is the recording software, not the program.

If needed, I have also implemented an imperfect auto-solving algorithm, one iteration of which is activateable by pressing the space button. Note that the algorithm cannot fully solve grids where there are multiple possible solutions.

<p align="center">
<img src="big.gif" width="450" height="450"/>
</p>
