from tiles.tile import Tile

def test_get_enum_from_id():
    expected = Tile.PACMAN
    actual = Tile.get_enum_from_id(2)
    assert actual == expected

def test_get_enum_from_symbol():
    expected = Tile.PACMAN
    actual = Tile.get_enum_from_symbol("c")
    assert actual == expected

def test_get_symbol_from_id():
    expected = "c"
    actual = Tile.get_symbol_from_id(2)
    assert actual == expected

def test_get_id_from_symbol():
    expected = 2
    actual = Tile.get_id_from_symbol("c")
    assert actual == expected

def test_str():
    assert str(Tile.PACMAN) == "PACMAN"

def test_repr():
    assert str(Tile.PACMAN) == "PACMAN"