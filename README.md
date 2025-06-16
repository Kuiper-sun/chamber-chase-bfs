# Chamber Chase 🧠🎮

**Chamber Chase** is a turn-based, tile-based puzzle and strategy game where players must outwit a smart AI pursuer in a confined grid chamber. The objective: reach the exit before getting caught.

---

## 🕹 Game Overview

Chamber Chase is inspired by classic grid-based puzzle games like *Deadly Rooms of Death (DROD)*, focusing on a tense one-on-one cat-and-mouse experience. The player must strategically navigate a grid while avoiding an AI that adapts based on visibility and player movement.

### 🎯 Objective
- **Win Condition**: Reach any exit tile before the AI catches you.
- **Lose Condition**: Get caught by the AI (AI and player on the same tile).
- **Move Limit**: Each game session has a maximum number of moves, dynamically calculated based on difficulty and map layout.

---

## ⚙ Game Rules

- **Grid**: Default 12x12 grid with empty tiles, obstacles, and exits.
- **Movement**: Player and AI move orthogonally (up/down/left/right).
- **Turn Order**: Player moves first, then the AI.
- **Vision**: Both entities have limited vision ranges, blocked by obstacles.
- **Obstacles**: Impassable and block line-of-sight.
- **AI States**:
  - **Patrol**: Moves between strategic patrol points.
  - **Chase**: Activated when the player is visible.
  - **Investigate**: Moves to the last known player position if vision is lost.
- **Difficulty Levels**: Affects grid size, number of exits, obstacle density, AI intelligence, and move limit.

---

## 🤖 AI Implementation

The AI combines a **state machine** with **Breadth-First Search (BFS)** for pathfinding:

- **Patrol**: Covers key areas and exits.
- **Chase**: Pursues the player aggressively when spotted.
- **Investigate**: Searches the last known player location for a few turns.
- **Detection**: AI uses a vision radius with line-of-sight logic.
- **Adaptive Behavior**: Difficulty level affects patrol paths, vision range, and intelligence.

---

## 🧱 Software Architecture

Chamber Chase is built using modular components for maintainability and scalability:

- **Game Engine**: Handles game loop, win/loss conditions, and turn logic.
- **Grid Manager**: Manages grid layout, tile states, and valid movements.
- **Player Controller**: Processes input and updates player state.
- **AI Controller**: Manages state transitions and calculates AI moves.
- **Vision System**: Calculates what each agent can see in real-time.
- **Difficulty Manager**: Dynamically adjusts game settings.
- **UI**: Displays grid, player/AI positions, and outcome messages.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- (Optional) A GUI library like `tkinter` or `pygame` if applicable

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Kuiper-sun/chamber-chase-bfs.git
   cd chamber-chase-bfs
