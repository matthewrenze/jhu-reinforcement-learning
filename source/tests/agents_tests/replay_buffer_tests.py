from agents.replay_buffer import ReplayBuffer

def test_init():
    replay_buffer = ReplayBuffer(1000)
    assert len(replay_buffer.buffer) == 0
    assert replay_buffer.buffer.maxlen == 1000

def test_add():
    replay_buffer = ReplayBuffer(1000)
    replay_buffer.add(1, 2, 3, 4, False)
    assert len(replay_buffer.buffer) == 1
    assert replay_buffer.buffer[0] == (1, 2, 3, 4, False)

def test_sample():
    replay_buffer = ReplayBuffer(1000)
    replay_buffer.add(1, 2, 3, 4, False)
    states, actions, rewards, next_states, dones = replay_buffer.sample(1)
    assert states == (1,)
    assert actions == (2,)
    assert rewards == (3,)
    assert next_states == (4,)
    assert dones == (False,)

# NOTE: Will this test produce different tuples each time?
# Should I convert to sets instead?
def test_sample_multiple():
    replay_buffer = ReplayBuffer(1000)
    replay_buffer.add(1, 2, 3, 4, False)
    replay_buffer.add(5, 6, 7, 8, True)
    states, actions, rewards, next_states, dones = replay_buffer.sample(2)
    assert states == (1, 5)
    assert actions == (2, 6)
    assert rewards == (3, 7)
    assert next_states == (4, 8)
    assert dones == (False, True)

