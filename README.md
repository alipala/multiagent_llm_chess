# Chess Game Project

## Project Overview

This project implements an AI-powered chess game using multiple agents, natural language processing, and machine learning techniques. The game allows two AI agents to play chess against each other, with moves determined by a combination of chess engine analysis and language model reasoning.

## 🌟 Features

- Multi-agent system with specialized roles:
  - White Player Agent: Aggressive style with sound principles
  - Black Player Agent: Solid positional play focus
  - Board Proxy Agent: Manages game state and move validation
  - Commentator Agent: Provides engaging game analysis

- Advanced Chess Engine:
  - Position evaluation using material and positional factors
  - Opening principles enforcement
  - Tactical awareness
  - King safety evaluation

- Interactive Web Interface:
  - Real-time game visualization
  - Move validation
  - PGN export functionality
  - Game state management via WebSocket
  - Detailed game analysis and commentary

- RAG (Retrieval Augmented Generation):
  - Integrated chess knowledge base
  - Position-specific strategy retrieval
  - Dynamic game analysis

## 🔧 Technical Stack

- **Backend Framework**: Flask with SocketIO
- **AI/ML Components**:
  - AutoGen for multi-agent orchestration
  - LangChain for RAG implementation
  - OpenAI GPT-4 for strategic decision making
  - Custom Chess Engine for position evaluation

- **Libraries**:
  - python-chess: Core chess logic
  - ChromaDB: Vector store for chess knowledge
  - PyTorch: Neural network operations
  - Eventlet: Async operations
  
## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chess-ai-project
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 💻 Usage

1. Start the server:
```bash
python app.py
```

2. Access the application:
```
Local: http://localhost:5001
Network: http://0.0.0.0:5001
```
## 🎮 Game Features

### Agent System
- **White Player Agent**: 
  - Aggressive style
  - Focuses on tactical opportunities
  - Follows sound opening principles

- **Black Player Agent**:
  - Positional play style
  - Emphasis on solid pawn structure
  - Strategic piece placement

- **Board Proxy Agent**:
  - Manages game state
  - Validates moves
  - Tracks game progress

- **Commentator Agent**:
  - Provides move analysis
  - Offers strategic insights
  - Explains key positions

### Chess Engine Features
- Material evaluation
- Positional assessment
- Opening principles enforcement
- King safety evaluation
- Tactical awareness
- Move validation system

### Web Interface Features
- Real-time board updates
- Move validation
- Game state management
- PGN export
- Game analysis display
- WebSocket communication

## 🔍 Code Structure

```
├── app.py                 # Main application file
├── models/
│   └── base_model.py     # Chess engine implementation
├── static/               # Frontend assets
├── templates/           # HTML templates
└── requirements.txt     # Project dependencies
```

## 🛠️ API Endpoints

- `/`: Main chess interface
- `/export_pgn`: PGN export endpoint

### WebSocket Events
- `connect`: Initial connection
- `make_move`: Handle player moves
- `request_ai_move`: Request AI move
- `reset_game`: Reset game state
- `get_game_summary`: Get game analysis
- `get_pgn`: Export game in PGN format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request