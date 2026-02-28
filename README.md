# Game_524 - Q-Learning Gunfight (Simplified)

## Setup
- Python 3.9+ recommended
- Install pygame:
  `pip install pygame`
- Optional for reward curve plot:
  `pip install matplotlib`

## Train (creates Q-table + logs)
`python main.py --train`

Outputs:
- `models/qtable.json`
- `logs/train_rewards.csv`
- `logs/train_rewards.png` (if matplotlib installed)

## Play
Human vs AI:
`python main.py --mode human`

AI vs scripted opponent:
`python main.py --mode ai`

## Notes
- Controls: WASD move, Space shoot, R reset, Enter to start.
- The HUD shows the AI state and chosen action for report screenshots.
