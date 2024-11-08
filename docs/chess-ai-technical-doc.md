# Chess AI System: Technical Documentation

## Table of Contents
1. [AI Architecture](#ai-architecture)
2. [Core AI Components](#core-ai-components)
3. [Multi-Agent System](#multi-agent-system)
4. [RAG Implementation](#rag-implementation)
5. [Move Generation System](#move-generation-system)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling & Recovery](#error-handling--recovery)

## AI Architecture

### 1. Hybrid Neural-Symbolic System

#### Base Language Model (GPT-4)
```python
class GPT4Model:
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        
    def encode_position(self, board: chess.Board) -> str:
        return f"""
        Position FEN: {board.fen()}
        Material count: {self._get_material_count(board)}
        King safety: {self._analyze_king_safety(board)}
        Center control: {self._analyze_center_control(board)}
        """
```

#### Chess-Specific Transformer
```python
class ChessTransformer(nn.Module):
    def __init__(self, encoder_layers: int = 6, attention_heads: int = 8, d_model: int = 512):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, d_model)
        self.position_embedding = nn.Embedding(64, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, attention_heads),
            encoder_layers
        )
```

### 2. Integration Architecture
```python
class HybridArchitecture:
    def __init__(self, llm: GPT4Model, chess_transformer: ChessTransformer):
        self.llm = llm
        self.chess_transformer = chess_transformer
        self.piece_to_index = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
            '.': 0, None: 0
        }
```

## Core AI Components

### 1. Position Evaluation
```python
def evaluate_position(self, board: chess.Board) -> float:
    try:
        material_score = self._evaluate_material(board)
        position_score = self._evaluate_positional_factors(board)
        
        total_score = (
            0.6 * material_score +
            0.4 * position_score
        )
        
        return max(min(total_score / 100, 10), -10)
    except Exception as e:
        logger.error(f"Error in position evaluation: {str(e)}")
        return 0.0
```

### 2. Move Generation
```python
def get_move(self, board: chess.Board, legal_moves: List[str]) -> Tuple[str, str]:
    try:
        if not legal_moves:
            raise ValueError("No legal moves available")

        move_scores = {}
        for move in legal_moves:
            scores = [self._evaluate_move(board, move) for _ in range(3)]
            move_scores[move] = sum(scores) / 3

        top_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        selected_move = random.choice(top_moves)[0]
        
        return selected_move, self._generate_explanation(board, selected_move)
    except Exception as e:
        return random.choice(legal_moves), "Move selected randomly due to error"
```

## Multi-Agent System

### 1. Agent Configuration
```python
player_system_message = """
You are a chess player with deep knowledge of chess principles.
Before making a move:
1. Check if the game is over
2. Get legal moves
3. Use get_best_move for move selection
4. Make the move using UCI format
5. Explain your reasoning
End with 'Your move.' to prompt the other player.
"""

player_white = ConversableAgent(
    name="Player_White",
    system_message=player_system_message,
    llm_config=llm_config,
)
```

### 2. Function Registration
```python
for caller in [player_white, player_black]:
    register_function(is_game_over, caller=caller, executor=board_proxy)
    register_function(make_move, caller=caller, executor=board_proxy)
    register_function(get_best_move, caller=caller, executor=board_proxy)
    register_function(get_legal_moves, caller=caller, executor=board_proxy)
```

## RAG Implementation

### 1. Knowledge Base Setup
```python
chess_knowledge = """
# Opening Theory
1. Basic Principles:
   - Control the center
   - Develop pieces early
   - Castle for king safety
   
2. Common Openings:
   - Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
   - Sicilian Defense: 1.e4 c5
   - Queen's Gambit: 1.d4 d5 2.c4
"""

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(chess_knowledge)
docsearch = Chroma.from_texts(texts, embeddings)
```

### 2. Query System
```python
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o"),
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3})
)
```

## Move Generation System

### 1. Evaluation Pipeline
```python
def _evaluate_move(self, board: chess.Board, move: str) -> float:
    try:
        chess_move = chess.Move.from_uci(move)
        score = 0.0
        
        # Base randomization
        score += random.uniform(0, 0.2)
        
        # Piece value evaluation
        piece = board.piece_at(chess_move.from_square)
        if piece:
            piece_values = {'P': 1, 'N': 3, 'B': 3.2, 'R': 5, 'Q': 9, 'K': 0}
            score += piece_values.get(piece.symbol().upper(), 0) * 0.1

        # Position evaluation
        to_square = chess_move.to_square
        if to_square in central_squares:
            score += 0.3 + random.uniform(0, 0.4)
            
        return score
    except Exception as e:
        return random.uniform(0, 1)
```

### 2. Move Selection Logic
```python
def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str, float]:
    try:
        board = chess.Board(board_fen)
        move, explanation = hybrid_brain.get_move(board, legal_moves)
        
        board_copy = board.copy()
        chess_move = chess.Move.from_uci(move)
        board_copy.push(chess_move)
        evaluation = hybrid_brain.evaluate_position(board_copy)
        
        return move, explanation, evaluation
    except Exception as e:
        return legal_moves[0], "Fallback move selected", 0.0
```

## Performance Optimization

### 1. Memory Management
```python
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.after_request
def after_request(response):
    cleanup_memory()
    return response
```

### 2. Error Recovery
```python
@socketio.on_error()
def handle_error(e):
    logger.error(f"WebSocket error: {str(e)}")
    emit('error', {
        'message': 'An error occurred',
        'details': str(e)
    })

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"HTTP error: {str(e)}")
    return {
        'success': False,
        'error': str(e)
    }, 500
```

## Error Handling & Recovery

### 1. Graceful Degradation
```python
def handle_ai_move():
    try:
        if game_over or ai_thinking:
            return

        ai_thinking = True
        best_move, explanation, evaluation = get_best_move(
            board.fen(), 
            [m.uci() for m in board.legal_moves]
        )
        
        if not best_move:
            raise ValueError("No valid move generated")
            
    except Exception as e:
        logger.error(f"Error in AI move: {str(e)}")
        emit('error', {'message': str(e)})
    finally:
        ai_thinking = False
```

### 2. State Recovery
```python
def handle_reset_game():
    try:
        global board, move_count, game_over, game_tracker
        board = chess.Board()
        move_count = 0
        game_over = False
        game_tracker = GameTracker(evaluator=hybrid_brain)
        
        emit('game_state', {
            'fen': board.fen(),
            'legal_moves': [move.uci() for move in board.legal_moves]
        })
    except Exception as e:
        logger.error(f"Error resetting game: {str(e)}")
        emit('error', {'message': str(e)})
```

This technical documentation provides a detailed overview of the AI implementation in the chess system. Each section includes actual code snippets and explanations of the underlying logic and design decisions.