#!/usr/bin/env python3

from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


@dataclass
class BlockSize:
    """
    x, y, z are positive integers representing the exponent of a power of two.
        x + y + z <= 10
    and, of course
        (2**x)*(2**y)*(2**z)
    """

    x: int
    y: int
    z: int


@dataclass
class EncodedBlockSize:
    """
    size (int): 0 = small, 1 = medium, 2 = big
    shape (int): 0 = plane on z, 1 = cube
    """

    size: int
    shape: int


def inbounds(block: BlockSize):
    if block.x < 0 or block.y < 0 or block.z < 0:
        return False

    return (block.x + block.y + block.z) <= 10


def from_block_size(block: BlockSize):
    #todo
    return EncodedBlockSize(block.x, block.y, block.z)


def to_block_size(encoded: EncodedBlockSize):
    #todo
    return BlockSize(encoded.a, encoded.b, encoded.c)


def main():
    final: EncodedBlockSize

    BLK_SIZE = 0 # 0,1,2 => small, medium, big

    
    SHAPE = 0 # 0 is a "plane" on z, 2 is a perfect cube 



if __name__ == "__main__":
    main()
