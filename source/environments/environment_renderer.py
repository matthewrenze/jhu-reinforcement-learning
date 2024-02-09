import os
import numpy as np
from environments.states import Tile
from colorama import Fore, Style, init

tile_colors = {
    Tile.EMPTY: Fore.BLACK,
    Tile.WALL: Fore.BLUE,
    Tile.PACMAN: "\033[93;1m",
    Tile.GHOST: "\033[91;1m",
    Tile.DOT: "\033[97m",
    Tile.POWER: "\033[97m",
    Tile.BONUS: Fore.GREEN
}

init()

def refresh_screen(data, lines):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(data)
    print(f"\033[{lines}A", end="")

def draw_screen(environment, total_reward):
    lines = 0
    data = "\n"
    state = environment.get_state()
    for row in state:
        for tile_id in np.nditer(row):
            tile_id = int(tile_id)
            tile_symbol = Tile.get_symbol_from_id(tile_id)
            tile_enum = Tile.get_enum_from_id(tile_id)
            tile_color = tile_colors[tile_enum]
            tile_text = tile_color + tile_symbol + "  " + Style.RESET_ALL
            data += tile_text
        data += "\n"
        lines += 1
    data += f"Total Reward: {total_reward}"
    return data, lines

def render(environment, total_reward):
    data, lines = draw_screen(environment, total_reward)
    refresh_screen(data, lines)

# def render(environment, total_reward):
#     state = environment.get_state()
#     for row in state:
#         for tile_id in np.nditer(row):
#             tile_id = int(tile_id)
#             tile_symbol = Tile.get_symbol_from_id(tile_id)
#             tile_enum = Tile.get_enum_from_id(tile_id)
#             tile_color = tile_colors[tile_enum]
#             tile_text = tile_color + tile_symbol + "  " + Style.RESET_ALL
#             print(tile_text, end="")
#         print()
#     print()
#     print(f"Total Reward: {total_reward}")