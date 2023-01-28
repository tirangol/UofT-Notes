"""Infinity Loop - Video Game

The program should automatically run upon launch. Otherwise, the following code can be run:

>>> game_grid = Grid()
>>> game_grid.game()
"""
from __future__ import annotations
from typing import Optional
import random
import math
import pygame

BACKGROUND_COL = (23, 42, 46)
BLOCK_COL = (107, 157, 169)
TEXT_COL = (255, 255, 255)


class Block:
    """A block in Infinity Loop.

    Instance Attributes:
    - up: whether the block points up
    - down: whether the block points down
    - left: whether the block points left
    - right: whether the block points right
    - solved: whether the block is in the solved position (by the solving algorithm)
    - d: the block's direction (1 - 4)

    Representation Invariants:
    - 1 <= self.d <= 4
    """
    up: bool
    down: bool
    left: bool
    right: bool
    solved: bool
    d: int

    def __init__(self) -> None:
        """Initialize the block."""
        self.solved = False
        self.d = random.randint(1, 4)
        self.orient_to_d()

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        raise NotImplementedError

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        raise NotImplementedError

    def is_aligned_up(self, up: Optional[Block]) -> bool:
        """Return whether the block is in an aligned position upwards."""
        if up is None:
            return not self.up
        return self.up == up.down

    def is_aligned_down(self, down: Optional[Block]) -> bool:
        """Return whether the block is in an aligned position downwards."""
        if down is None:
            return not self.down
        return self.down == down.up

    def is_aligned_left(self, left: Optional[Block]) -> bool:
        """Return whether the block is in an aligned position leftwards."""
        if left is None:
            return not self.left
        return self.left == left.right

    def is_aligned_right(self, right: Optional[Block]) -> bool:
        """Return whether the block is in an aligned position rightwards."""
        if right is None:
            return not self.right
        return self.right == right.left

    def rotate(self) -> None:
        """Rotate the block 90 degrees clockwise."""
        self.up, self.down, self.left, self.right = self.left, self.right, self.down, self.up
        self.update_d()

    def align(self, up: bool, down: bool, left: bool, right: bool) -> None:
        """Align the block a certain direction."""
        self.up, self.down, self.left, self.right = up, down, left, right

    def __str__(self) -> str:
        """Return a string representation of the block."""
        raise NotImplementedError


class Empty(Block):
    """An empty block with zero nodes."""

    def __init__(self) -> None:
        """Initialize the block."""
        Block.__init__(self)
        self.up, self.down, self.left, self.right = False, False, False, False
        self.solved = True

    def __str__(self) -> str:
        """Return a string representation of the block."""
        return " "

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        pass

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        pass


class One(Block):
    """A block with one node.
    """

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        self.up = True if self.d == 1 else False
        self.down = True if self.d == 3 else False
        self.left = True if self.d == 4 else False
        self.right = True if self.d == 2 else False
        if self.d >= 5 or self.d <= 0:
            raise BlockError('One', -1, -1)

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        if self.up:
            self.d = 1
        elif self.right:
            self.d = 2
        elif self.down:
            self.d = 3
        elif self.left:
            self.d = 4
        else:
            raise BlockError('One', -1, -1)

    def __str__(self) -> str:
        """Return a string representation of the block."""
        if self.up:
            return "^"
        elif self.down:
            return "v"
        elif self.left:
            return "<"
        else:
            return ">"


class Line(Block):
    """A line block with two nodes."""

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        self.up = True if self.d in [1, 3] else False
        self.down = True if self.d in [1, 3] else False
        self.left = True if self.d in [2, 4] else False
        self.right = True if self.d in [2, 4] else False
        if self.d >= 5 or self.d <= 0:
            raise BlockError('Line', -1, -1)

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        if self.up and self.down:
            self.d = 1
        elif self.left and self.right:
            self.d = 2
        else:
            raise BlockError('Line', -1, -1)

    def __str__(self) -> str:
        """Return a string representation of the block."""
        if self.left:
            return "─"
        else:
            return "│"


class Corner(Block):
    """A corner block with two nodes."""

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        self.up = True if self.d in [1, 2] else False
        self.down = True if self.d in [3, 4] else False
        self.left = True if self.d in [4, 1] else False
        self.right = True if self.d in [2, 3] else False
        if self.d >= 5 or self.d <= 0:
            raise BlockError('Corner', -1, -1)

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        if self.up and self.left:
            self.d = 1
        elif self.up and self.right:
            self.d = 2
        elif self.right and self.down:
            self.d = 3
        elif self.left and self.down:
            self.d = 4
        else:
            raise BlockError('Corner', -1, -1)

    def __str__(self) -> str:
        """Return a string representation of the block."""
        if self.down:
            if self.left:
                return "┐"
            else:
                return "┌"
        else:
            if self.left:
                return "┘"
            else:
                return "└"


class Tri(Block):
    """A block with three nodes."""

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        self.up = False if self.d == 1 else True
        self.down = False if self.d == 3 else True
        self.left = False if self.d == 4 else True
        self.right = False if self.d == 2 else True
        if self.d >= 5 or self.d <= 0:
            raise BlockError('Tri', -1, -1)

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        if not self.up:
            self.d = 1
        elif not self.right:
            self.d = 2
        elif not self.down:
            self.d = 3
        elif not self.left:
            self.d = 4
        else:
            raise BlockError('Tri', -1, -1)

    def __str__(self) -> str:
        """Return a string representation of the block."""
        if not self.up:
            return "┬"
        elif not self.down:
            return "┴"
        elif not self.left:
            return "├"
        else:
            return "┤"


class Full(Block):
    """An full block with four nodes."""

    def __init__(self) -> None:
        """Initialize the block."""
        Block.__init__(self)
        self.up, self.down, self.left, self.right = True, True, True, True
        self.solved = True

    def __str__(self) -> str:
        """Return a string representation of the block."""
        return "┼"

    def orient_to_d(self) -> None:
        """Orient the block depending on self.d."""
        pass

    def update_d(self) -> None:
        """Update self.d to match the block's direction as indicated by self.up/down/left/right."""
        pass


class Grid:
    """A grid in Infinity Loop.

    Instance Attributes:
    - rows: the number of rows in the grid
    - cols: the number of columns in the grid
    - grid: the grid string all the game information
    """
    rows: int
    cols: int
    grid: list[list[Optional[Block]]]

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        """Initialize the grid."""
        self.rows = rows
        self.cols = cols

        each_row = []
        for _ in range(cols):
            each_row.append(None)

        self.grid = []
        for _ in range(rows):
            self.grid.append(each_row.copy())

    def __str__(self) -> str:
        """Return a string representation of the grid."""
        str_so_far = ''
        for row in self.grid:
            for x in row:
                str_so_far += x.__str__() + ' '
            str_so_far += '\n'
        return str_so_far

    def last_row(self, i: int) -> bool:
        """Return whether the object is in the grid's last row."""
        return i == self.rows - 1

    def last_col(self, j: int) -> bool:
        """Return whether the object is in the grid's last column."""
        return j == self.cols - 1

    def check(self) -> bool:
        """Return whether every object in the grid is aligned."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if i == 0:
                    if self.grid[i][j].up:
                        return False
                    check_up = True
                else:
                    check_up = self.grid[i][j].is_aligned_up(self.grid[i - 1][j])

                if self.last_row(i):
                    if self.grid[i][j].down:
                        return False
                    check_down = True
                else:
                    check_down = self.grid[i][j].is_aligned_down(self.grid[i + 1][j])

                if j == 0:
                    if self.grid[i][j].left:
                        return False
                    check_left = True
                else:
                    check_left = self.grid[i][j].is_aligned_left(self.grid[i][j - 1])

                if self.last_col(j):
                    if self.grid[i][j].right:
                        return False
                    check_right = True
                else:
                    check_right = self.grid[i][j].is_aligned_right(self.grid[i][j + 1])

                if not (check_up and check_down and check_left and check_right):
                    return False
        return True

    def up_empty(self, i: int, j: int) -> bool:
        """Check if the above object is empty or a boundary."""
        return i == 0 or (i != 0 and isinstance(self.grid[i - 1][j], Empty))

    def left_empty(self, i: int, j: int) -> bool:
        """Check if the left object is empty or a boundary."""
        return j == 0 or (j != 0 and isinstance(self.grid[i][j - 1], Empty))

    def down_empty(self, i: int, j: int) -> bool:
        """Check if the below object is empty or a boundary."""
        if self.last_row(i):
            return True
        return isinstance(self.grid[i + 1][j], Empty)

    def right_empty(self, i: int, j: int) -> bool:
        """Check if the right object is empty or a boundary."""
        if self.last_col(j):
            return True
        return isinstance(self.grid[i][j + 1], Empty)

    def solve_corners(self) -> None:
        """Arrange in the correct order all the grid objects that are touching boundaries."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                e_up = self.up_empty(i, j)
                e_down = self.down_empty(i, j)
                e_left = self.left_empty(i, j)
                e_right = self.right_empty(i, j)
                e_total = sum([e_up, e_down, e_left, e_right])

                # One
                if isinstance(self.grid[i][j], One):
                    if e_total == 3:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not e_up, not e_down, not e_left, not e_right)
                    if e_total >= 4:
                        raise BlockError('One', i, j)
                # Line
                elif isinstance(self.grid[i][j], Line):
                    if 1 <= e_total <= 2:
                        self.grid[i][j].solved = True
                        if e_total == 1:
                            e_up = e_down = e_up or e_down
                            e_left = e_right = e_left or e_right
                        self.grid[i][j].align(not e_up, not e_down, not e_left, not e_right)
                    if e_total >= 3:
                        raise BlockError('Line', i, j)
                # Corner
                elif isinstance(self.grid[i][j], Corner):
                    if e_total == 2 and ((e_up and e_right) or (e_right and e_down) or
                                         (e_down and e_left) or (e_left and e_up)):
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not e_up, not e_down, not e_left, not e_right)
                    if e_total >= 3:
                        raise BlockError('Corner', i, j)
                # Tri
                elif isinstance(self.grid[i][j], Tri):
                    if e_total == 1:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not e_up, not e_down, not e_left, not e_right)
                    if e_total >= 2:
                        raise BlockError('Tri', i, j)

                self.grid[i][j].update_d()

    def up_solved(self, i: int, j: int) -> bool:
        """Check if the above object is not a boundary, solved and pointed downwards."""
        return i != 0 and self.grid[i - 1][j].solved and self.grid[i - 1][j].down

    def down_solved(self, i: int, j: int) -> bool:
        """Check if the below object is not a boundary, solved and pointed upwards."""
        if self.last_row(i):
            return False
        return self.grid[i + 1][j].solved and self.grid[i + 1][j].up

    def left_solved(self, i: int, j: int) -> bool:
        """Check if the left object is not a boundary, solved and pointed rightwards."""
        return j != 0 and self.grid[i][j - 1].solved and self.grid[i][j - 1].right

    def right_solved(self, i: int, j: int) -> bool:
        """Check if the right object is not a boundary, solved and pointed leftwards."""
        if self.last_col(j):
            return False
        return self.grid[i][j + 1].solved and self.grid[i][j + 1].left

    def up_invalid(self, i: int, j: int) -> bool:
        """Check if the above object is either solved and pointed wrongly, or empty."""
        return self.up_empty(i, j) or (self.grid[i - 1][j].solved and not self.grid[i - 1][j].down)

    def left_invalid(self, i: int, j: int) -> bool:
        """Check if the left object is either solved and pointed wrongly, or empty."""
        return self.left_empty(i, j) or (
                self.grid[i][j - 1].solved and not self.grid[i][j - 1].right)

    def down_invalid(self, i: int, j: int) -> bool:
        """Check if the below object is either solved and pointed wrongly, or empty."""
        return self.down_empty(i, j) or (self.grid[i + 1][j].solved and not self.grid[i + 1][j].up)

    def right_invalid(self, i: int, j: int) -> bool:
        """Check if the right object is either solved and pointed wrongly, or empty."""
        return self.right_empty(i, j) or (
                self.grid[i][j + 1].solved and not self.grid[i][j + 1].left)

    def solve_step(self) -> None:
        """Iterate once through the grid, arranging all its objects in the correct order."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                i_up = self.up_invalid(i, j)
                i_down = self.down_invalid(i, j)
                i_left = self.left_invalid(i, j)
                i_right = self.right_invalid(i, j)
                i_total = sum([i_up, i_down, i_left, i_right])

                s_up = self.up_solved(i, j)
                s_down = self.down_solved(i, j)
                s_left = self.left_solved(i, j)
                s_right = self.right_solved(i, j)
                s_total = sum([s_up, s_down, s_left, s_right])

                # One
                if isinstance(self.grid[i][j], One):
                    # 1 adjacent block is solved, pointed correctly
                    if s_total == 1:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(s_up, s_down, s_left, s_right)
                    # All adjacent blocks except one are invalid
                    elif i_total == 3:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not i_up, not i_down, not i_left, not i_right)
                    elif s_total >= 2 or i_total >= 4:
                        raise BlockError('One', i, j)
                # Line
                elif isinstance(self.grid[i][j], Line):
                    # 1 or 2 adjacent blocks are solved, pointed correctly
                    if 1 <= s_total <= 2:
                        if s_total == 1 or (
                                s_total == 2 and ((s_up and s_down) or (s_left and s_right))):
                            self.grid[i][j].solved = True
                            s_up = s_down = s_up or s_down
                            s_left = s_right = s_left or s_right
                            self.grid[i][j].align(s_up, s_down, s_left, s_right)

                            if (i_up and s_down) or (i_left and s_right) or (i_down and s_up) or (
                                    s_left and i_right):
                                raise BlockError('Line', i, j)
                        else:
                            raise BlockError('Line', i, j)
                    # 1 or 2 adjacent blocks are invalid
                    elif 1 <= i_total <= 2:
                        if i_total == 1 or (
                                i_total == 2 and ((i_up and i_down) or (i_left and i_right))):
                            self.grid[i][j].solved = True
                            i_up = i_down = i_up or i_down
                            i_left = i_right = i_left or i_right
                            self.grid[i][j].align(not i_up, not i_down, not i_left, not i_right)
                        else:
                            raise BlockError('Line', i, j)
                    elif s_total >= 3 or i_total >= 3:
                        raise BlockError('Line', i, j)
                # Corner
                elif isinstance(self.grid[i][j], Corner):
                    # 1 block is invalid, the other is solved
                    if i_total == 1 and s_total == 1:
                        if (i_up and s_right) or (i_left and s_down):
                            self.grid[i][j].solved = True
                            self.grid[i][j].align(False, True, False, True)
                        elif (i_up and s_left) or (i_right and s_down):
                            self.grid[i][j].solved = True
                            self.grid[i][j].align(False, True, True, False)
                        elif (i_down and s_right) or (i_left and s_up):
                            self.grid[i][j].solved = True
                            self.grid[i][j].align(True, False, False, True)
                        elif (i_down and s_left) or (i_right and s_up):
                            self.grid[i][j].solved = True
                            self.grid[i][j].align(True, False, True, False)
                    # 2 adjacent blocks in nearby directions are solved, pointed correctly
                    elif s_total == 2 and ((s_up and s_right) or (s_right and s_down) or
                                           (s_down and s_left) or (s_left and s_up)):
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(s_up, s_down, s_left, s_right)
                    # 2 adjacent blocks in nearby directions are invalid
                    elif i_total == 2 and ((i_up and i_right) or (i_right and i_down) or (
                            i_down and i_left) or (i_left and i_up)):
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not i_up, not i_down, not i_left, not i_right)
                    elif s_total >= 3 or i_total >= 3 or (i_up and i_down) or (
                            i_left and i_right) or (s_up and s_down) or (s_left and s_right):
                        raise BlockError('Corner', i, j)
                # Tri
                elif isinstance(self.grid[i][j], Tri):
                    # 3 adjacent blocks are solved, pointed correctly
                    if s_total == 3:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(s_up, s_down, s_left, s_right)
                    # 1 adjacent block is invalid
                    elif i_total == 1:
                        self.grid[i][j].solved = True
                        self.grid[i][j].align(not i_up, not i_down, not i_left, not i_right)
                    elif s_total >= 4 or i_total >= 2:
                        raise BlockError('Tri', i, j)

                self.grid[i][j].update_d()

    def solve(self) -> None:
        """Attempt to solve the grid."""
        self.solve_corners()
        counter = 0

        temp = self.grid.copy()
        self.solve_step()

        while temp != self.grid:
            if counter == 1000:
                break
            temp = self.grid.copy()
            self.solve_step()
            counter += 1

    def brute_force(self) -> None:
        """Attempt to solve the grid with brute force."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if not self.grid[i][j].solved and not isinstance(self.grid[i][j], (Empty, Full)):
                    self.grid[i][j].d = random.choice(self.rotations(i, j))
                    self.grid[i][j].orient_to_d()
                    self.grid[i][j].solved = True
                    self.solve()
                    return

    def rotations(self, i: int, j: int) -> list:
        """Return the possible orientations of an object in the grid."""
        possible = [1, 2, 3, 4]
        if isinstance(self.grid[i][j], One):
            # up = 1, right = 2, down = 3, left = 4
            if self.up_invalid(i, j):
                possible.remove(1)
            if self.right_invalid(i, j):
                possible.remove(2)
            if self.down_invalid(i, j):
                possible.remove(3)
            if self.left_invalid(i, j):
                possible.remove(4)
        elif isinstance(self.grid[i][j], Line):
            # up/down = 1/3, left/right = 2/4
            if self.up_invalid(i, j) or self.down_invalid(i, j):
                possible.remove(1)
                possible.remove(3)
            if self.left_invalid(i, j) or self.right_invalid(i, j):
                possible.remove(2)
                possible.remove(4)
        elif isinstance(self.grid[i][j], Corner):
            # up/left = 1, up/right = 2, down/right = 3, down/left = 4
            if self.up_invalid(i, j) or self.left_invalid(i, j):
                possible.remove(1)
            if self.up_invalid(i, j) or self.right_invalid(i, j):
                possible.remove(2)
            if self.down_invalid(i, j) or self.right_invalid(i, j):
                possible.remove(3)
            if self.down_invalid(i, j) or self.left_invalid(i, j):
                possible.remove(4)
        elif isinstance(self.grid[i][j], Tri):
            # down = 1, left = 2, up = 3, right = 4
            if self.down_invalid(i, j):
                possible.remove(1)
            if self.left_invalid(i, j):
                possible.remove(2)
            if self.up_invalid(i, j):
                possible.remove(3)
            if self.right_invalid(i, j):
                possible.remove(4)
        return possible

    def environment(self, i: int, j: int) -> set:
        """Return the objects that can be added at a certain position in the grid."""
        if i != 0 and self.grid[i - 1][j] is None:
            s_up, i_up = False, False
        else:
            s_up = self.up_solved(i, j)
            i_up = self.up_invalid(i, j)

        if j != 0 and self.grid[i][j - 1] is None:
            s_left, i_left = False, False
        else:
            s_left = self.left_solved(i, j)
            i_left = self.left_invalid(i, j)

        if self.last_row(i):
            s_down = self.down_solved(i, j)
            i_down = self.down_invalid(i, j)
        else:
            if self.grid[i + 1][j] is None:
                s_down, i_down = False, False
            else:
                s_down = self.down_solved(i, j)
                i_down = self.down_invalid(i, j)

        if self.last_col(j):
            s_right = self.right_solved(i, j)
            i_right = self.right_invalid(i, j)
        else:
            if self.grid[i][j + 1] is None:
                s_right, i_right = False, False
            else:
                s_right = self.right_solved(i, j)
                i_right = self.right_invalid(i, j)

        s_total = sum([s_up, s_down, s_left, s_right])
        i_total = sum([i_up, i_down, i_left, i_right])

        # 4 sides connect to it
        if s_total == 4:
            objects = {4}
        # 3 sides connect to it
        elif s_total == 3:
            objects = {3, 4}
        # 2 sides connect to it
        elif s_total == 2:
            # 2 connecting sides are opposite
            if (s_up and s_down) or (s_left and s_right):
                objects = {2, 3, 4}
            # 2 connecting sides are adjacent
            else:
                objects = {2.5, 3, 4}
        # 1 side connects to it
        elif s_total == 1:
            objects = {1, 2, 2.5, 3, 4}
        # 0 sides connect to it
        else:
            objects = {0, 1, 2, 2.5, 3, 4}
        # 0 = empty, 1 = one, 2 = line, 2.5 = corner, 3 = tri, 4 = full

        # 1+ side doesn't connect to it
        if i_total >= 1:
            objects = objects.difference({4})

            # 1+ side doesn't connect, 1-2 sides connect to it
            if 1 <= s_total <= 2:
                objects = objects.difference({0})
                # A connecting and non-connecting side are opposite (no line)
                if (i_up and s_down) or (i_left and s_right) or (i_right and s_left) or (
                        i_down and s_up):
                    objects = objects.difference({2})
                # Two non-connecting sides are opposite (no corner)
                elif (s_up and s_down) or (s_left and s_right):
                    objects = objects.difference({2.5})
            # 1+ side doesn't connect, 3 sides connect to it
            elif s_total == 3:
                return {3}

            # 2+ sides don't connect
            if i_total >= 2:
                objects = objects.difference({3})

                # Two non-connecting sides are opposite (no corner)
                if (i_up and i_down) or (i_left and i_right):
                    objects = objects.difference({2.5})
                # Two non-connecting sides are adjacent (no line)
                else:
                    objects = objects.difference({2})

                # 3+ sides don't connect
                if i_total >= 3:
                    objects = objects.difference({2, 2.5})

                    # 4 sides don't connect
                    if i_total == 4:
                        objects = objects.difference({1})
        return objects

    def auto_align(self, i: int, j: int, counter: int = 5) -> None:
        """Auto-align a block in the grid to fit already-put-in-place blocks during generation."""
        # If above is empty
        if i == 0:
            # Not pointing up
            c1 = not self.grid[i][j].up
            c5 = True
        # If above is None (ignore)
        elif self.grid[i - 1][j] is None:
            c1, c5 = True, True
        else:
            # Above is invalid -> not pointing up
            c1 = not self.up_invalid(i, j) or not self.grid[i][j].up
            # Above is solved -> pointing up
            c5 = not self.up_solved(i, j) or self.grid[i][j].up

        # If left is empty
        if j == 0:
            # Not pointing left
            c3 = not self.grid[i][j].left
            c7 = True
        # If above is None (ignore)
        elif self.grid[i][j - 1] is None:
            c3, c7 = True, True
        else:
            # Left is invalid -> not pointing left
            c3 = not self.left_invalid(i, j) or not self.grid[i][j].left
            # Left is solved -> pointing left
            c7 = not self.left_solved(i, j) or self.grid[i][j].left

        # If below is empty
        if self.last_row(i):
            # Not pointing down
            c2 = not self.grid[i][j].down
            c6 = True
        else:
            # If below is None (ignore)
            if self.grid[i + 1][j] is None:
                c2, c6 = True, True
            else:
                # Down is invalid -> not pointing down
                c2 = not self.down_invalid(i, j) or not self.grid[i][j].down
                # Down is solved -> pointing down
                c6 = not self.down_solved(i, j) or self.grid[i][j].down

        # If right is empty
        if self.last_col(j):
            # Not pointing right
            c4 = not self.grid[i][j].right
            c8 = True
        else:
            # If right is None (ignore)
            if self.grid[i][j + 1] is None:
                c4, c8 = True, True
            else:
                # Right is invalid -> not pointing right
                c4 = not self.right_invalid(i, j) or not self.grid[i][j].right
                # Right is solved -> pointing right
                c8 = not self.right_solved(i, j) or self.grid[i][j].right

        if not all([c1, c2, c3, c4, c5, c6, c7, c8]) and counter >= 0:
            self.grid[i][j].rotate()
            self.auto_align(i, j, counter - 1)

    def generate(self) -> None:
        """Generate a game to play."""
        self.clear()
        spaces = self.rows * self.cols
        coords = [(x, y) for x in range(self.rows) for y in range(self.cols)]

        while spaces > 0:
            x, y = coords.pop(random.randint(0, len(coords) - 1))
            if len(self.environment(x, y)) == 0:
                raise GenerateError()
            else:
                get_object = random.choice(list(self.environment(x, y)))
                if get_object == 0:
                    get_object = Empty()
                elif get_object == 1:
                    get_object = One()
                elif get_object == 2:
                    get_object = Line()
                elif get_object == 2.5:
                    get_object = Corner()
                elif get_object == 3:
                    get_object = Tri()
                elif get_object == 4:
                    get_object = Full()
                else:
                    raise GenerateError()
            self.grid[x][y] = get_object
            self.auto_align(x, y)
            self.grid[x][y].solved = True
            spaces -= 1

    def scramble(self) -> None:
        """Scramble the grid."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if isinstance(self.grid[i][j], Empty):
                    self.grid[i][j] = Empty()
                elif isinstance(self.grid[i][j], One):
                    self.grid[i][j] = One()
                elif isinstance(self.grid[i][j], Line):
                    self.grid[i][j] = Line()
                elif isinstance(self.grid[i][j], Corner):
                    self.grid[i][j] = Corner()
                elif isinstance(self.grid[i][j], Tri):
                    self.grid[i][j] = Tri()
                elif isinstance(self.grid[i][j], Full):
                    self.grid[i][j] = Full()
        if self.check():
            self.scramble()

    def clear(self) -> None:
        """Clear the grid."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.grid[i][j] = None

    ################################################################################################
    # Version of the game using only pygame rectangles and circles
    ################################################################################################
    def game(self, screen_size: tuple[int, int] = (800, 800)) -> None:
        """Open up a solveable game version of the grid."""
        self.generate()
        self.scramble()
        pygame.display.init()
        pygame.display.set_caption("Infinity Loop")
        pygame.font.init()
        screen = pygame.display.set_mode(screen_size)
        screen.fill(BACKGROUND_COL)
        pygame.display.flip()

        pygame.event.clear()
        pygame.event.set_blocked(None)
        pygame.event.set_allowed([pygame.QUIT, pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN])

        x = min((screen_size[0] - 100) / self.cols, (screen_size[1] - 100) / self.rows, 100)
        origin = (screen_size[0] - (self.cols * x)) / 2, (screen_size[1] - (self.rows * x)) / 2
        win = False
        meaningless_key = False

        while True:
            screen.fill(BACKGROUND_COL)
            pygame.display.flip()
            if not meaningless_key:
                self.draw_grid(screen, x, origin)
                pygame.display.flip()
            if win:
                font = pygame.font.SysFont("Tahoma", min(screen_size[0], screen_size[1]) // 20)
                text = font.render("You Win!", True, TEXT_COL)
                screen.blit(text, (screen_size[0] // 2.5, 10))
                pygame.display.flip()

            meaningless_key = False
            event = pygame.event.wait()

            if win:
                self.generate()
                self.scramble()
                win = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                win = self.handle_click(x, origin, event)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.solve()
                    if self.check():
                        win = True
            elif event.type == pygame.QUIT:
                break
            else:
                meaningless_key = True

        pygame.display.quit()
        exit()

    def draw_grid(self, screen: pygame.Surface, x: float, origin: tuple[float, float]) -> None:
        """Draw the grid in the game version."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.draw_block(screen, self.grid[i][j],
                                (round(j * x + origin[0]), round(i * x + origin[1])), x)

    def draw_vertical_half_line(self, screen: pygame.Surface, start: tuple[float, float],
                                x: float, offset: bool) -> None:
        """Draw a vertical half-line in the game version."""
        if offset:
            pygame.draw.line(screen, BLOCK_COL, (start[0] + (x / 2), start[1] + (x / 2)),
                             (start[0] + (x / 2), start[1] + x), round(x / 10))
        else:
            pygame.draw.line(screen, BLOCK_COL, (start[0] + (x / 2), start[1]),
                             (start[0] + (x / 2), start[1] + (x / 2)), round(x / 10))

    def draw_horizontal_half_line(self, screen: pygame.Surface, start: tuple[float, float],
                                  x: float, offset: bool) -> None:
        """Draw a horizontal half-line in the game version."""
        if offset:
            pygame.draw.line(screen, BLOCK_COL,
                             (start[0] + (x / 2), start[1] + (x / 2)),
                             (start[0] + x, start[1] + (x / 2)), round(x / 10))
        else:
            pygame.draw.line(screen, BLOCK_COL, (start[0], start[1] + (x / 2)),
                             (start[0] + (x / 2), start[1] + (x / 2)), round(x / 10))

    def draw_circle(self, screen: pygame.Surface, position: tuple[float, float], x: float) -> None:
        """Draw a center circle in the game version."""
        pygame.draw.circle(screen, BLOCK_COL, (position[0] + (x / 2),
                                               position[1] + (x / 2)), x / 4)
        pygame.draw.circle(screen, BACKGROUND_COL, (position[0] + (x / 2),
                                                    position[1] + (x / 2)), x / 6)

    def draw_block(self, screen: pygame.Surface, block: Block, position: tuple[float, float],
                   x: float) -> None:
        """Draw a block in the game version."""
        if block.up:
            self.draw_vertical_half_line(screen, position, x, False)
        if block.right:
            self.draw_horizontal_half_line(screen, position, x, True)
        if block.down:
            self.draw_vertical_half_line(screen, position, x, True)
        if block.left:
            self.draw_horizontal_half_line(screen, position, x, False)
        if isinstance(block, One):
            self.draw_circle(screen, position, x)
        elif not isinstance(block, Empty):
            pygame.draw.circle(screen, BLOCK_COL, (position[0] + (x / 2),
                                                   position[1] + (x / 2)), x / 10)

    def handle_click(self, x: float, origin: tuple[float, float],
                     event: pygame.event.Event) -> bool:
        """Handle a mouse click event in the game version."""
        if origin[0] <= event.pos[0] <= origin[0] + (x * self.cols) and \
                origin[1] <= event.pos[1] <= origin[1] + (x * self.rows):
            i = math.floor((event.pos[1] - origin[1]) / x)
            j = math.floor((event.pos[0] - origin[0]) / x)
            try:
                self.grid[i][j].rotate()
            except IndexError:
                pass
            return True if self.check() else False


class BlockError(Exception):
    """The exception raised when an Infinity Loop object is connected incorrectly to surroundings.

    Instance Attributes:
    - block_type: the name of the block type causing the error.
    - row: the row of the block type causing the error.
    - col: the column of the block type causing the error
    """
    block_type: str
    row: int
    col: int

    def __init__(self, block_type: str, row: int, col: int) -> None:
        """Initialize this exception."""
        self.block_type = block_type
        self.row = row
        self.col = col

    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.row == -1:
            return 'A block of type ' + self.block_type + ' was initialized improperly.'
        else:
            return 'The block of type ' + self.block_type + ' at row ' + str(self.row) + \
                   ', column ' + str(self.col) + ' is improperly attached.'


class GenerateError(Exception):
    """The exception raised when the level generation process goes wrong."""

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return 'Something went wrong during the level generation process.'


# def example1() -> Grid:
#     """Generate an example grid."""
#     grid = Grid(5, 5)
#     grid.grid[0] = [One(), Line(), Corner(), Empty(), Empty()]
#     grid.grid[1] = [One(), One(), Tri(), Corner(), Empty()]
#     grid.grid[2] = [Line(), Empty(), Corner(), Full(), Corner()]
#     grid.grid[3] = [Tri(), Line(), Corner(), Line(), Line()]
#     grid.grid[4] = [Corner(), Line(), One(), One(), One()]
#     return grid
#
#
# def example2() -> Grid:
#     """Generate an example grid."""
#     grid = Grid(5, 5)
#     grid.grid[0] = [One(), One(), Empty(), Empty(), Empty()]
#     grid.grid[1] = [One(), One(), One(), One(), Empty()]
#     grid.grid[2] = [Empty(), One(), Tri(), Tri(), One()]
#     grid.grid[3] = [One(), One(), Tri(), Tri(), One()]
#     grid.grid[4] = [One(), Empty(), One(), One(), Empty()]
#     return grid


if __name__ == "__main__":
    gridx, gridy = 10, 10
    screenx, screeny = 800, 800
    print()
    print("Infinity Loop - Pygame Version (by Richard Yin)")
    print("Type in the commands below:")
    print()
    print("grid x y   - sets the game grid size to x by y (default = 10 x 10, min grid area = 8)")
    print("screen x y - sets the screen size to x by y (default = 800 x 800, min = 50 x 50)")
    print("play       - begin the Infinity Loop game")
    print()
    print("If you press space in-game, the program makes 1 step in an attempt to solve the puzzle.")
    inp = input().lower()
    while "  " in inp:
        inp = inp.replace("  ", " ")

    while True:
        while all(x not in inp for x in {"grid", "screen", "play"}):
            print("Unrecognized input. Please try again.")
            inp = input()

        if inp == "play":
            grid = Grid(gridx, gridy)
            grid.generate()
            grid.game((screenx, screeny))
            break
        else:
            if inp.count(" ") != 2:
                print("Invalid argument count. Please try again.")
            command, newx, newy = inp.split(" ")
            newx, newy = round(float(newx)), round(float(newy))
            if command == "grid":
                if newx * newy >= 8:
                    gridx, gridy = newx, newy
                    print("Successfully changed grid size to " + str(newx) + " by " + str(newy))
                else:
                    print("Error: grid size is too small.")
            else:
                if newx >= 50 and newy >= 50:
                    screenx, screeny = newx, newy
                    print("Successfully changed screen size to " + str(newx) + " by " + str(newy))
                else:
                    print("Error: screen size is too small")
        inp = input()
