import copy
import numpy as np
import random

class Disc:
    NONE = "."
    BLACK = "x"
    WHITE = "o"
    WALL = "#"
    OPPOSITE = {BLACK: WHITE, WHITE: BLACK}

class Board:
    DIRECTIONS_STEP = (-11, -10, -9, -1, 1, 9, 10, 11)

    def __init__(self):
        _, X, O, W = Disc.NONE, Disc.BLACK, Disc.WHITE, Disc.WALL
        self.discs = [
            W, W, W, W, W, W, W, W, W, W,
            W, _, _, _, _, _, _, _, _, W,
            W, _, _, _, _, _, _, _, _, W,
            W, _, _, _, _, _, _, _, _, W,
            W, _, _, _, O, X, _, _, _, W,
            W, _, _, _, X, O, _, _, _, W,
            W, _, _, _, _, _, _, _, _, W,
            W, _, _, _, _, _, _, _, _, W,
            W, _, _, _, _, _, _, _, _, W,
            W, W, W, W, W, W, W, W, W, W
        ]
        self.places = [i for i, d in enumerate(self.discs) if d == Disc.NONE]

    def putable_places(self, disc):
        return [p for p in self.places if self.reverse_places(disc, p)]

    def put(self, disc, place):
        flips = self.reverse_places(disc, place)
        if not flips:
            raise ValueError(f"Cannot put {disc} at {place}")
        self.discs[place] = disc
        for f in flips:
            self.discs[f] = disc
        self.places.remove(place)

    def reverse_places(self, disc, place):
        opp = Disc.OPPOSITE[disc]
        if disc not in (Disc.BLACK, Disc.WHITE) or place not in self.places:
            return []
        flips = []
        for d in self.DIRECTIONS_STEP:
            line = []
            p = place + d
            while 0 <= p < len(self.discs) and self.discs[p] == opp:
                line.append(p)
                p += d
            if line and 0 <= p < len(self.discs) and self.discs[p] == disc:
                flips.extend(line)
        return flips

    def is_pass(self, disc):
        return not any(self.reverse_places(disc, p) for p in self.places)

    def count(self, disc):
        return self.discs.count(disc)

    def is_game_over(self):
        return not self.places or (self.is_pass(Disc.BLACK) and self.is_pass(Disc.WHITE))

    def show(self):
        for i in range(1, 9):
            row = self.discs[i*10+1 : i*10+9]
            print(' '.join(row))
        print()

def get_state_tensor(board: Board, agent_disc: str):
    """(3,8,8) テンソル：
       0: agent の石、
       1: 相手の石、
       2: 合法手マスク
    """
    tensor = np.zeros((3, 8, 8), dtype=np.float32)
    opp_disc = Disc.OPPOSITE[agent_disc]

    # 中央8×8を走査
    for i in range(8):
        for j in range(8):
            idx = (i+1)*10 + (j+1)
            if board.discs[idx] == agent_disc:
                tensor[0, i, j] = 1.0
            elif board.discs[idx] == opp_disc:
                tensor[1, i, j] = 1.0

    # 合法手マスク
    for p in board.putable_places(agent_disc):
        i, j = divmod(p, 10)
        tensor[2, i-1, j-1] = 1.0

    return tensor  # shape=(3,8,8)

class OthelloEnv:
    def __init__(self, agent_disc=Disc.BLACK):
        self.agent_disc = agent_disc
        self.reset()

    def reset(self):
        self.board = Board()
        self.current_player = Disc.BLACK
        self.opponent = Disc.WHITE
        return self.get_state()

    def step(self, action):
        # action=None => pass
        if action is not None:
            self.board.put(self.current_player, action)

        # 終了判定
        done = self.board.is_game_over()

        # 報酬計算（agent_disc 視点）
        if done:
            b = self.board.count(Disc.BLACK)
            w = self.board.count(Disc.WHITE)
            if b > w:
                winner = Disc.BLACK
            elif w > b:
                winner = Disc.WHITE
            else:
                winner = None

            if winner == self.agent_disc:
                reward = 1.0
            elif winner is None:
                reward = 0.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        # プレイヤー交代（パス処理は step 内で自動）
        self.current_player, self.opponent = self.opponent, self.current_player
        if not done and self.board.is_pass(self.current_player):
            # 相手も置けないなら終局扱い、さもなくばもう一度交代
            if self.board.is_pass(self.opponent):
                done = True
            else:
                self.current_player, self.opponent = self.opponent, self.current_player

        return self.get_state(), reward, done, {}

    def get_state(self):
        return get_state_tensor(self.board, self.current_player)

    def legal_actions(self):
        return self.board.putable_places(self.current_player)

    def render(self):
        self.board.show()

    def legal_actions(self):
        return self.board.putable_places(self.current_player)

    def legal_actions_opp(self):
        """対戦相手の合法手リストを返す"""
        return self.board.putable_places(self.opponent)

# 動作テスト
if __name__ == "__main__":
    env = OthelloEnv(agent_disc=Disc.BLACK)
    state = env.reset()
    env.render()
    print("合法手:", env.legal_actions())
    # ランダムプレイ1手
    action = random.choice(env.legal_actions())
    s2, r, done, _ = env.step(action)
    print("Action:", action, "Reward:", r, "Done:", done)
    env.render()
