import os
import argparse
import numpy as np
from tqdm import trange
import torch
import random

from osero_for_Q import OthelloEnv, Disc
from dqn import DQNAgent, DEVICE

def parse_args():
    parser = argparse.ArgumentParser("Othello DQN Training")
    parser.add_argument("--episodes", type=int, default=20000,
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
            # 現状態テンソル化
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # 合法手リスト（10×10表記）
            legal_env = env.legal_actions()

            if not legal_env:
                # パス
                next_state, _, done, _ = env.step(None)
                reward = 0.0  # パス時は報酬なし
            else:
                # 1) 環境idx → フラット idx
                legal_idx = [pos_to_index(p) for p in legal_env]

                # 2) DQN が 0–63 の中から選択
                action_idx = agent.select_action(state_tensor, legal_idx)

                # 3) フラット idx → 環境idx に戻す
                action_env = index_to_pos(action_idx)

                # フォールバック：もし action_env が合法手リストにない場合はランダム選択
                if action_env not in legal_env:
                    action_env = random.choice(legal_env)

                # 実際に盤面を進める
                next_state, _, done, _ = env.step(action_env)

                # ─── 報酬シェーピング ───
                my_disc  = env.agent_disc
                opp_disc = Disc.OPPOSITE[my_disc]
                b = env.board.count(my_disc)
                w = env.board.count(opp_disc)

                corners = [11, 18, 81, 88]
                my_corners  = sum(env.board.discs[c] == my_disc  for c in corners)
                opp_corners = sum(env.board.discs[c] == opp_disc for c in corners)

                mobility_diff = len(env.legal_actions()) - len(env.legal_actions_opp())

                reward = (b - w) / 64.0 \
                       + 0.5 * (my_corners - opp_corners) \
                       + 0.1 * mobility_diff
                # ───────────────────────

                # 経験リプレイ用に記憶＆学習
                next_tensor = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)
                agent.store_transition(
                    state_tensor,
                    action_idx,
                    reward,
                    next_tensor,
                    done
                )
                agent.update()
                ep_reward += reward

            state = next_state

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
