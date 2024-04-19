def find_legal_positions(tiles, current_position):
    legal_positions = []
    all_positions = [current_position]
    height = tiles.shape[0]
    width = tiles.shape[1]

    transitions = [(-1,0),(1,0),(0,-1),(0,1)]
    for i in transitions: 
        new_x = current_position[0] + i[0]
        new_y = current_position[1] + i[1]
        new_location = (new_x, new_y)

        if _can_teleport(new_location, height, width):
            new_location = _teleport(new_location, height, width)
    
        if tiles[new_location] != 1:
            legal_positions.append(new_location)
        else: 
            new_location = current_position
        
        all_positions.append(new_location)

    return legal_positions, all_positions

def find_legal_actions(tiles, current_position): 
    transitions = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
    legal_positions = find_legal_positions(tiles, current_position)[0]
    legal_positions.append((current_position[0],current_position[1]))
    legal_actions = []
    for i in range(len(transitions)): 
        delta = transitions[i]
        if (current_position[0]+delta[0], current_position[1]+delta[1]) in legal_positions: 
            legal_actions.append(i)
    return legal_actions

def _can_teleport(new_location: tuple[int, int], height, width) -> bool:
    if new_location[0] < 0 \
            or new_location[1] < 0 \
            or new_location[0] >= height \
            or new_location[1] >= width:
        return True
    return False


def _teleport(new_location: tuple[int, int], height, width) -> tuple[int, int]:
    if new_location[0] < 0:
        new_location = (height - 1, new_location[1])
    if new_location[0] >= height:
        new_location = (0, new_location[1])
    if new_location[1] < 0:
        new_location = (new_location[0], width - 1)
    if new_location[1] >= width:
        new_location = (new_location[0], 0)
    return new_location

    
        
