from actions.action import Action

def test_str():
    action = Action.UP
    assert str(action) == "UP"