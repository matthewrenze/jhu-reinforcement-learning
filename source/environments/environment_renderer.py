import os
import numpy as np
from tiles.tile import Tile
from colorama import Fore, Style, init

tile_colors = {
    Tile.EMPTY: Fore.BLACK,
    Tile.WALL: "\033[38;5;20m",
    Tile.PACMAN: "\033[38;5;226m",
    Tile.DOT: "\033[97m",
    Tile.POWER: "\033[97m",
    # Tile.BONUS: Fore.GREEN,
    # Tile.GHOST: "\033[91;1m",
    Tile.STATIC: Fore.WHITE,
    Tile.BLINKY: "\033[38;5;196m",
    Tile.PINKY: "\033[38;5;219m",
    Tile.INKY: "\033[38;5;51m",
    Tile.CLYDE: "\033[38;5;208m",
}

ghost_ids = [
    Tile.STATIC.id,
    Tile.BLINKY.id,
    Tile.PINKY.id,
    Tile.INKY.id,
    Tile.CLYDE.id]

init()

def refresh_screen(data, lines):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(data)
    print(f"\033[{lines}A", end="")

def draw_screen(environment, total_reward):
    lines = 0
    data = "\n"
    state = environment.get_state().tiles
    for row in state:
        for tile_id in np.nditer(row):
            tile_id = tile_id.item()
            tile_symbol = Tile.get_symbol_from_id(tile_id)
            tile_enum = Tile.get_enum_from_id(tile_id)
            tile_color = tile_colors[tile_enum]
            if tile_id == Tile.PACMAN.id:
                if environment._invincible_time > 0:
                    tile_symbol = "C"
                if environment.is_game_over and not environment.is_winner:
                    tile_color = Fore.WHITE
            if tile_id in ghost_ids:
                tile_symbol = "m"
                if environment._invincible_time > 0:
                    tile_color = "\033[38;5;33m"
            tile_text = tile_color + tile_symbol + "  " + Style.RESET_ALL
            data += tile_text
        data += "\n"
        lines += 1
    # TODO: Need to render Pacman after ghosts so that Pacman is on top
    data += f"Total Reward: {total_reward}\n"
    if environment._invincible_time > 0:
        data += f"Invincibility: {environment._invincible_time}\n"
    if environment.is_game_over:
        data += "Game Over. "
        if environment.is_winner:
            data += "You win!"
        else:
            data += "You lose."
    return data, lines

def render(environment, total_reward):
    data, lines = draw_screen(environment, total_reward)
    refresh_screen(data, lines)