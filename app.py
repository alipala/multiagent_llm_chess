import os
import chess
import chess.svg
import chess.pgn
from typing import List, Union, Tuple, Optional
from typing_extensions import Annotated
import openai
import torch
import torch.nn as nn
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from autogen import ConversableAgent, register_function, config_list_from_json
from dotenv import load_dotenv
from datetime import datetime
import time
from io import StringIO
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging
import warnings
from typing import List, Tuple 
import sys
from models.base_model import GPT4Model, ChessTransformer, HybridArchitecture

def log_startup_status():
    """Log detailed startup status"""
    logger.info("="*50)
    logger.info("Chess AI Application Startup Status")
    logger.info("="*50)
    logger.info("1. Server Configuration:")
    logger.info(f"   - Host: 0.0.0.0")
    logger.info(f"   - Port: 5001")
    logger.info(f"   - Debug Mode: False")
    logger.info("2. Access URLs:")
    logger.info(f"   - Local: http://localhost:5001")
    logger.info(f"   - Network: http://0.0.0.0:5001")
    logger.info("3. Available Endpoints:")
    logger.info("   - / (Main chess interface)")
    logger.info("   - /export_pgn (PGN export)")
    logger.info("4. WebSocket Events Ready:")
    logger.info("   - connect")
    logger.info("   - make_move")
    logger.info("   - request_ai_move")
    logger.info("   - reset_game")
    logger.info("   - get_game_summary")
    logger.info("   - get_pgn")
    logger.info("="*50)
    logger.info("Server is running and ready for connections!")
    logger.info("="*50)


# Suppress specific Autogen warnings
warnings.filterwarnings("ignore", message="Function .* is being overridden", category=UserWarning)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
# Suppress specific loggers
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("autogen").setLevel(logging.ERROR)

# Initialize Flask and SocketIO
app = Flask(__name__, static_url_path='/static')
app.config['SERVER_NAME'] = None 
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai.api_key

if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

logger.info(f"API key loaded: {openai.api_key[:5]}...{openai.api_key[-5:]}")

# Initialize AI models
def initialize_models():
    try:
        logger.info("Initializing AI models...")
        gpt4_model = GPT4Model(temperature=0.7)
        chess_transformer = ChessTransformer()
        hybrid_brain = HybridArchitecture(
            llm=gpt4_model,
            chess_transformer=chess_transformer
        )
        logger.info("AI models initialized successfully")
        return hybrid_brain
    except Exception as e:
        logger.error(f"Error initializing AI models: {str(e)}")
        raise

# Initialize hybrid brain
try:
    hybrid_brain = initialize_models()
    logger.info("Hybrid brain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize hybrid brain: {str(e)}")
    raise

config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

logger.info(f"Config list: {config_list}")

# Chess knowledge base
chess_knowledge = """
# Opening Theory
1. Basic Principles:
   - Control the center
   - Develop pieces early
   - Castle for king safety
   - Connect rooks
   
2. Common Openings:
   - Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
   - Sicilian Defense: 1.e4 c5
   - Queen's Gambit: 1.d4 d5 2.c4
   - King's Indian Defense: 1.d4 Nf6 2.c4 g6

# Middlegame Strategy
1. Piece Activity:
   - Knights on outposts
   - Bishops on open diagonals
   - Rooks on open files
   - Queen behind development

2. Pawn Structure:
   - Isolated pawns
   - Doubled pawns
   - Backward pawns
   - Passed pawns

# Endgame Principles
1. Basic Checkmates:
   - King and Queen vs King
   - King and Rook vs King
   - King and Two Bishops vs King

2. Pawn Endgames:
   - Opposition
   - Square rule
   - Breakthrough
   - Key squares

# Tactical Patterns
1. Common Motifs:
   - Pin
   - Fork
   - Skewer
   - Discovery
   - Double attack

2. Calculation:
   - Forcing moves
   - Candidate moves
   - Move order
   - Tempo

# Strategic Concepts
1. Position Assessment:
   - Material balance
   - King safety
   - Pawn structure
   - Piece activity

2. Plan Formation:
   - Weaknesses
   - Piece placement
   - Pawn breaks
   - Attack preparation
"""

# Set up RAG system
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(chess_knowledge)
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])

# Create a retriever with a limit on the number of results
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Set up the RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o"),
    chain_type="stuff",
    retriever=retriever
)

# Game state variables
board = chess.Board()
made_move = False
move_count = 0
game_over = False

# Game Tracker
class GameTracker:
    def __init__(self):
        self.moves = []
        self.captures = []
        self.checks = []
        self.castlings = []
        self.material_balance = []
        self.position_scores = []
        self.time_per_move = []
        self.logger = logging.getLogger(__name__)

    def add_move(self, board: chess.Board, move: chess.Move, time_taken: float = 0.0):
        """
        Track a new move and its implications
        """
        try:
            # Record basic move info
            san_move = board.san(move)
            self.moves.append(san_move)
            self.time_per_move.append(time_taken)
            
            # Track captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                self.captures.append((san_move, captured_piece))
                self.logger.info(f"Capture recorded: {san_move} takes {captured_piece}")
            
            # Track checks
            if board.is_check():
                self.checks.append(san_move)
                self.logger.info(f"Check recorded: {san_move}")
            
            # Track castling
            if board.is_castling(move):
                self.castlings.append(san_move)
                self.logger.info(f"Castling recorded: {san_move}")
            
            # Calculate material balance
            material = self._calculate_material_balance(board)
            self.material_balance.append(material)
            
            # Calculate position score
            score = self._evaluate_position(board)
            self.position_scores.append(score)
            
            self.logger.info(f"Move {san_move} fully processed and tracked")
            
        except Exception as e:
            self.logger.error(f"Error tracking move: {str(e)}")
            raise

    def _calculate_material_balance(self, board: chess.Board) -> int:
        """
        Calculate material balance from White's perspective
        """
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        balance = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.symbol().upper()]
                balance += value if piece.color == chess.WHITE else -value
                
        return balance

    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Simple position evaluation
        """
        material = self._calculate_material_balance(board)
        center_control = self._evaluate_center_control(board)
        king_safety = self._evaluate_king_safety(board)
        
        return material + 0.1 * center_control + 0.2 * king_safety

    def _evaluate_center_control(self, board: chess.Board) -> int:
        """
        Evaluate center control
        """
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = 0
        
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                control += 1
            if board.is_attacked_by(chess.BLACK, square):
                control -= 1
                
        return control

    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """
        Evaluate king safety
        """
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        white_safety = len(list(board.attackers(chess.BLACK, white_king_square)))
        black_safety = len(list(board.attackers(chess.WHITE, black_king_square)))
        
        return black_safety - white_safety

    def get_statistics(self) -> dict:
        """
        Get game statistics
        """
        return {
            'total_moves': len(self.moves),
            'captures': len(self.captures),
            'checks': len(self.checks),
            'castlings': len(self.castlings),
            'average_time': sum(self.time_per_move) / len(self.time_per_move) if self.time_per_move else 0,
            'material_trajectory': self.material_balance,
            'position_scores': self.position_scores
        }

game_tracker = GameTracker()

# Game functions
def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str]:
    """
    Get the best move using the hybrid AI system
    """
    try:
        board = chess.Board(board_fen)
        move, explanation = hybrid_brain.get_move(board, legal_moves)
        logger.info(f"Generated move: {move} with explanation")
        return move, explanation
    except Exception as e:
        logger.error(f"Error in get_best_move: {str(e)}")
        fallback_move = legal_moves[0] if legal_moves else None
        if not fallback_move:
            raise ValueError("No legal moves available")
        return fallback_move, "Fallback move selected due to error"

def get_legal_moves() -> str:
    """Get all legal moves in current position"""
    return "Possible moves are: " + ",".join([move.uci() for move in board.legal_moves])

def make_move(move: str, explanation: str = "") -> Tuple[str, str, bool]:
    """
    Make a move on the board
    Returns: (result_message, explanation, is_game_over)
    """
    global made_move, board, move_count, game_over, game_tracker
    
    if game_over:
        return "The game is already over.", explanation, True

    try:
        start_time = time.time()
        chess_move = chess.Move.from_uci(move)
        
        if chess_move in board.legal_moves:
            game_tracker.add_move(board, chess_move, time.time() - start_time)
            board.push(chess_move)
            made_move = True
            move_count += 1

            piece = board.piece_at(chess_move.to_square)
            piece_symbol = piece.symbol() if piece else ''
            piece_name = chess.piece_name(piece.piece_type) if piece else ''

            result = f"Moved {piece_name} ({piece_symbol}) from "\
                     f"{chess.SQUARE_NAMES[chess_move.from_square]} to "\
                     f"{chess.SQUARE_NAMES[chess_move.to_square]}."

            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                captured_piece_name = chess.piece_name(captured_piece.piece_type)
                result += f" Captured {captured_piece_name}."

            if board.is_check():
                result += " Check!"

            game_over = is_game_over()
            if game_over:
                if board.is_checkmate():
                    result += f" Checkmate! {'White' if board.turn == chess.BLACK else 'Black'} wins."
                elif board.is_stalemate():
                    result += " Stalemate! The game is a draw."
                elif board.is_insufficient_material():
                    result += " Draw due to insufficient material."
                elif board.is_seventyfive_moves():
                    result += " Draw due to seventy-five moves rule."
                elif board.is_fivefold_repetition():
                    result += " Draw due to fivefold repetition."
                else:
                    result += " The game is over."

            logger.info(f"Move made: {result}")
            logger.info(f"Explanation: {explanation}")
            return result, explanation, game_over
        else:
            logger.warning(f"Illegal move attempted: {move}")
            return f"Illegal move: {move}. Legal moves are: {get_legal_moves()}", explanation, game_over
    except ValueError:
        logger.error(f"Invalid move format: {move}")
        return f"Invalid move format: {move}. Please use UCI format (e.g., 'e2e4', 'g1f3').", explanation, game_over

def is_game_over() -> bool:
    """Check if the game is over"""
    global game_over
    game_over = game_over or board.is_game_over() or move_count >= 100  # 50 moves per player
    return game_over

def check_made_move(msg: str) -> bool:
    """Check if a move was made and update game status"""
    global made_move, game_over
    if made_move:
        made_move = False
        game_over = is_game_over()
    return game_over

def summarize_game(tracker: GameTracker, result: str, total_moves: int) -> str:
    """
    Generate a comprehensive game summary
    """
    try:
        summary_prompt = f"""
        Analyze this chess game:
        Total moves: {total_moves}
        Result: {result}
        Opening moves: {', '.join(tracker.moves[:5])}
        Key statistics:
        - Captures: {len(tracker.captures)}
        - Checks: {len(tracker.checks)}
        - Castling moves: {', '.join(tracker.castlings)}
        - Material balance trajectory: {tracker.material_balance}
        - Position scores: {tracker.position_scores}

        Recent captures: {', '.join([f"{move} (captured {piece})" for move, piece in tracker.captures[-3:]])}
        Recent checks: {', '.join(tracker.checks[-3:])}
        """

        summary = qa.run(summary_prompt)
        logger.info("Game summary generated successfully")
        return summary
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        return f"Error generating summary: {str(e)}"
    
def export_game_to_pgn(board: chess.Board, white_name: str = "Player_White", black_name: str = "Player_Black") -> str:
    """
    Export the game to PGN format
    """
    try:
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "AutoGen Chess Game"
        game.headers["Site"] = "Python Flask App"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = board.result()

        pgn_string = StringIO()
        exporter = chess.pgn.FileExporter(pgn_string)
        game.accept(exporter)

        logger.info("Game exported to PGN successfully")
        return pgn_string.getvalue()
    except Exception as e:
        logger.error(f"Error exporting game to PGN: {str(e)}")
        raise

# Player setup
player_system_message = """
You are a chess player with deep knowledge of chess principles, powered by a hybrid AI system combining language understanding and chess-specific transformers.
Before making a move, check if the game is over by calling is_game_over().
If the game is not over:
1. Use the get_legal_moves function to get the list of legal moves.
2. Use the get_best_move function to determine the best move using the hybrid AI system.
3. Call make_move(move) to make the move. Use only the move in UCI format.
4. After a move is made, explain your reasoning based on chess principles.
5. End your message with 'Your move.' to prompt the other player.
If the game is over, respond with 'The game has ended.' and do not send any further messages.
"""

# Create player agents
player_white = ConversableAgent(
    name="Player_White",
    system_message=player_system_message,
    llm_config=llm_config,
)

player_black = ConversableAgent(
    name="Player_Black",
    system_message=player_system_message,
    llm_config=llm_config,
)

board_proxy = ConversableAgent(
    name="Board_Proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

# Function registration
for caller in [player_white, player_black]:
    register_function(is_game_over, caller=caller, executor=board_proxy, name="is_game_over", description="Check if the game is over.")
    register_function(make_move, caller=caller, executor=board_proxy, name="make_move", description="Make a move on the chess board.")
    register_function(get_best_move, caller=caller, executor=board_proxy, name="get_best_move", description="Get the best move based on hybrid AI analysis.")
    register_function(get_legal_moves, caller=caller, executor=board_proxy, name="get_legal_moves", description="Get a list of legal moves in the current position.")

# Register nested chats
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
            "summary_method": "last_msg",
            "silent": True,
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
            "summary_method": "last_msg",
            "silent": True,
        }
    ],
)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/export_pgn')
def export_pgn():
    try:
        pgn = export_game_to_pgn(board)
        return pgn, 200, {'Content-Type': 'text/plain', 'Content-Disposition': 'attachment; filename=chess_game.pgn'}
    except Exception as e:
        logger.error(f"Error exporting PGN: {str(e)}")
        return str(e), 500

@socketio.on('connect')
def handle_connect():
    emit('game_state', {
        'fen': board.fen(),
        'legal_moves': [move.uci() for move in board.legal_moves]
    })
    logger.info("Client connected")

@socketio.on('make_move')
def handle_make_move(data):
    try:
        move = data['move']
        result, explanation, is_game_over = make_move(move)
        emit('move_made', {
            'move': move,
            'result': result,
            'explanation': explanation,
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves],
            'game_over': is_game_over
        })
        logger.info(f"Move handled: {move}")
    except Exception as e:
        logger.error(f"Error handling move: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('request_ai_move')
def handle_ai_move():
    try:
        logger.info("AI move requested")
        best_move, explanation = get_best_move(board.fen(), [m.uci() for m in board.legal_moves])
        logger.info(f"AI selected move: {best_move}")
        result, explanation, game_over = make_move(best_move, explanation)
        
        emit('move_made', {
            'move': best_move,
            'result': result,
            'explanation': explanation,
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves],
            'game_over': game_over
        })
        
        if game_over:
            summary = summarize_game(game_tracker, board.result(), move_count)
            emit('game_summary', {'summary': summary})
            
    except Exception as e:
        logger.error(f"Error in AI move: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('reset_game')
def handle_reset_game():
    try:
        global board, move_count, game_over, game_tracker
        board = chess.Board()
        move_count = 0
        game_over = False
        game_tracker = GameTracker()
        emit('game_state', {
            'fen': board.fen(),
            'legal_moves': [move.uci() for move in board.legal_moves]
        })
        logger.info("Game reset successfully")
    except Exception as e:
        logger.error(f"Error resetting game: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('get_game_summary')
def handle_get_summary():
    try:
        summary = summarize_game(game_tracker, board.result(), move_count)
        emit('game_summary', {'summary': summary})
        logger.info("Game summary generated and sent")
    except Exception as e:
        logger.error(f"Error getting game summary: {str(e)}")
        emit('error', {'message': str(e)})


@socketio.on('stop_ai_game')
def handle_stop_game():
    """Handle stop AI game event"""
    try:
        logger.info("="*50)
        logger.info("Stop AI Game Event Triggered")
        logger.info("="*50)
        
        # Get current game state
        stats = game_tracker.get_statistics()
        
        # Log detailed game statistics
        logger.info("Game Statistics at Stop:")
        logger.info(f"Total Moves: {stats['total_moves']}")
        logger.info(f"Captures: {stats['captures']}")
        logger.info(f"Checks: {stats['checks']}")
        logger.info(f"Castling Moves: {stats['castlings']}")
        if stats['total_moves'] > 0:
            logger.info(f"Average Time per Move: {stats['average_time']:.2f} seconds")
        
        # Log current position
        logger.info(f"Final Position: {board.fen()}")
        
        # Emit stop event to client
        emit('game_stopped', {
            'message': 'AI game stopped successfully',
            'statistics': stats,
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves]
        })
        
        logger.info("AI game stopped successfully")
        logger.info("="*50)
        
    except Exception as e:
        error_msg = f"Error stopping AI game: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        emit('error', {'message': error_msg})

@socketio.on('get_pgn')
def handle_get_pgn():
    try:
        pgn = export_game_to_pgn(board)
        emit('pgn_data', {'pgn': pgn})
        logger.info("PGN data generated and sent")
    except Exception as e:
        logger.error(f"Error getting PGN: {str(e)}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    try:
        port = 5001
        host = '0.0.0.0'
        logger.info("Starting the Chess AI application")
        logger.info(f"Server will be available at http://localhost:{port}")
        logger.info("Please wait for server initialization...")
        
        # Log startup status
        log_startup_status()
        
        # Run the server
        socketio.run(app, debug=False, host=host, port=port, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise