import os
import argparse
import numpy as np
from tqdm import trange

import torch

from osero_for_Q import OthelloEnv, Disc
from dqn import DQNAgent, DEVICE

def parse_args():
    parser = argparse.ArgumentParser("Othello DQN Training")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="総エピソード数")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="チェックポイント保存ディレクトリ")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="何エピソードごとにモデルを保存するか")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="何エピソードごとに進捗を表示するか")
    parser.add_argument("--load", type=str, default=None,
                        help="事前学習済モデルのパス（省略可）")
    return parser.parse_args()

def pos_to_index(p):
    """10×10のマス（11～88） → 8×8フラット idx（0–63）"""
    row, col = divmod(p, 10)
    return (row - 1) * 8 + (col - 1)

def index_to_pos(idx):
    """8×8フラット idx（0–63） → 10×10 マス idx（11～88）"""
    row8, col8 = divmod(idx, 8)
    return (row8 + 1) * 10 + (col8 + 1)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 環境＆エージェント初期化
    env = OthelloEnv(agent_disc=Disc.BLACK)
    agent = DQNAgent(n_actions=64)
    if args.load:
        agent.load(args.load)
        print(f"Loaded model from {args.load}")

    win_count = 0
    episode_rewards = []

    for ep in trange(1, args.episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # 1) state → tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # 2) 環境の合法手(11～88) → 8×8フラット idx のリスト
            legal_env = env.legal_actions()        # 例: [11, 22, 33, …]
            legal_idx = [pos_to_index(p) for p in legal_env]

            # 3) エージェントは0–63の idx を返す
            action_idx = agent.select_action(state_tensor, legal_idx)

            # 4) flatted idx → 環境用 idx（11–88） に変換
            action_env = index_to_pos(action_idx)
            next_state, reward, done, _ = env.step(action_env)

            # 5) Experience を格納＆モデル更新
            next_tensor = torch.tensor(next_state,
                                       dtype=torch.float32,
                                       device=DEVICE)
            agent.store_transition(state_tensor,
                                   action_idx,   # ← 0–63 の idx
                                   reward,
                                   next_tensor,
                                   done)
            agent.update()

            state = next_state
            ep_reward += reward

        episode_rewards.append(ep_reward)
        if ep_reward > 0:
            win_count += 1

        # 定期ログ出力
        if ep % args.log_interval == 0:
            recent = episode_rewards[-args.log_interval:]
            avg_reward = np.mean(recent)
            win_rate = win_count / args.log_interval
            print(f"[Episode {ep:5d}]  AvgReward: {avg_reward:.2f}  WinRate: {win_rate:.2f}")
            win_count = 0

        # 定期モデル保存
        if ep % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"dqn_ep{ep}.pth")
            agent.save(path)

    # 最終モデル保存
    final_path = os.path.join(args.save_dir, "dqn_final.pth")
    agent.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
