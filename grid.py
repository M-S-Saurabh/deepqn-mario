import numpy as np
import matplotlib.pyplot as plt


ROWS = 10
COLUMNS = 10
INITIAL_POSITION = [0, 2]
FINAL_POSITION = [ROWS-1, COLUMNS-1]

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
DUR = "ur"
DUL = "ul"
DDR = "dr"
DDL = "dl"
ACTIONS = [UP, DOWN, LEFT, RIGHT, DUR, DUL, DDR, DDL]

ALPHA = 0.8
GAMMA = .5
N = 10


class Grid:
    def __init__(self):
        self.grid = -1*np.ones((ROWS, COLUMNS))
        self.grid[FINAL_POSITION[0], FINAL_POSITION[1]] = 0

        self.player = INITIAL_POSITION

    def reset(self):
        self.player = INITIAL_POSITION
        return INITIAL_POSITION

    def step(self, action):
        x, y = 0, 0
        if action == RIGHT:
            y = 1
        elif action == LEFT:
            y = -1
        elif action == UP:
            x = -1
        elif action == DOWN:
            x = 1
        elif action == DDR:
            x, y = 1, 1
        elif action == DUR:
            x, y = -1, 1
        elif action == DUL:
            x, y = -1, -1
        elif action == DDL:
            x, y = 1, -1
        else:
            raise ValueError(f"Urecognized action '{action}'")

        px, py = self.player
        new_px = min(max(0, px + x), COLUMNS - 1)
        new_py = min(max(0, py + y), ROWS - 1)
        if action in [DDR, DUR, DUL, DDL]:
            if new_px == px or new_py == py:
                new_px, new_py = px, py
        self.player = (new_px, new_py)
        reward = self.grid[new_px, new_py]
        terminal = new_px == FINAL_POSITION[0] and new_py == FINAL_POSITION[1]
        return self.player, reward, terminal

    def __str__(self):
        grid = self.grid.copy()
        px, py = self.player
        grid[px, py] = -666

        output = list()
        for row in grid:
            lst = []
            for el in row:
                lst.append("   p" if el ==-666 else str(el))
            output.append("\t".join(lst))
        s = "\n".join(output)
        return s


class Agent:
    def __init__(self):
        self.Q = -1*np.ones((ROWS, COLUMNS, len(ACTIONS)))
        self.Q_target = self.Q.copy()

    def act(self, state):
        x, y = state
        i = np.argmax(self.Q[x, y])
        action = ACTIONS[i]
        return i, action

    def q_update(self, state, action_id, reward, next_state, is_next_state_terminal):
        if is_next_state_terminal:
            target = reward
        else:
            nx, ny = next_state
            target = reward + GAMMA*self.Q_target[nx, ny, :].max()
        x, y = state
        delta = target - self.Q[x, y, action_id]
        self.Q[x, y, action_id] = self.Q[x, y, action_id] + ALPHA*delta

    def best_policy(self):
        actions = []
        rows, columns, _ = self.Q.shape
        for row in range(rows):
            row_actions = []
            for column in range(columns):
                action_id, action = self.act((row, column))
                row_actions.append(action)
            actions.append(  ", ".join(row_actions)  )
        s = "\n".join(actions)
        return s

    def copy(self):
        self.Q_target = self.Q.copy()


def train_agent():
    num_episodes = 2000
    agent = Agent()
    env = Grid()
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        is_next_state_terminal = False
        i = 1
        while not is_next_state_terminal:
            action_id, action = agent.act(state)
            next_state, reward, is_next_state_terminal = env.step(action)
            agent.q_update(state, action_id, reward, next_state, is_next_state_terminal)

            i += 1
            if i % N == 0:
                agent.copy()

            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)

    m = min(rewards)
    plt.plot([x/m for x in rewards])
    plt.show()
    return env, agent


def test_agent(env, agent):
    state = env.reset()
    print(env)
    print()
    terminal = False
    while not terminal:
        _, action = agent.act(state)
        state, _, terminal = env.step(action)
        print(env)
        print()

if __name__ == "__main__":
    env, agent = train_agent()
    test_agent(env, agent)

