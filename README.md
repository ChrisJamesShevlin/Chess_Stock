# Chess-in-a-flash 2.1

A fast FEN-based Stockfish position analyzer with instant win‐probability display.

> **Whatever colour you select under “I am:”**  
> - **Positive score** → good for *you*.  
> - **Win% Me:XX – Opp:YY** → probability you win vs. your opponent.

All the usual features remain: reset, best-move preview, click-to-move, board flip.

![image](https://github.com/user-attachments/assets/b508fe2b-bffe-447a-8274-793c345f8ffe)


---

## Features

- **FEN Input**: Paste any valid FEN string and hit **Load FEN** (or Enter).
- **Side Selector**: Choose Auto (per FEN), White, or Black under **I am:**.
- **Instant Eval**:  
  - Centipawn score (±cp) shown in “Eval”.  
  - Win% for you vs. opponent via a logistic curve (≈75% at +300 cp).
- **Best‐Move Preview**:  
  - Show Stockfish’s top line non‐destructively; reset to clear.
- **Click-to-Move**:  
  - Click a piece, then click a target square.
- **SAN Eval**:  
  - Enter a move in SAN, hit **Evaluate** to play it and re-evaluate.

---

## Installation

```bash
git clone https://github.com/your-username/chess-in-a-flash.git
cd chess-in-a-flash
