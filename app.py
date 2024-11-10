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
import gc
from pathlib import Path


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
    logger.info("   - /analyze (Position analysis)")
    logger.info("   - /export_pgn (PGN export)")
    logger.info("4. WebSocket Events Ready:")
    logger.info("   - connect")
    logger.info("   - make_move")
    logger.info("   - request_ai_move")
    logger.info("   - analyze_position")
    logger.info("   - get_game_summary")
    logger.info("   - get_pgn")
    logger.info("5. AI Components:")
    logger.info("   - Hybrid Brain initialized")
    logger.info("   - Time Control system ready")
    logger.info("   - Pattern Recognition active")
    logger.info("   - Strategic Planning enabled")
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

# socketio = SocketIO(app, cors_allowed_origins="*") //for local run
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)


# Load environment variables
load_dotenv()
# Local run
# openai.api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai.api_key
# if not openai.api_key:
#     raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
# logger.info(f"API key loaded: {openai.api_key[:5]}...{openai.api_key[-5:]}")

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Set API key for OpenAI and environment
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger.info(f"API key loaded: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")


# Initialize AI models
def initialize_models():
    """Initialize and configure AI models"""
    try:
        logger.info("Initializing AI models...")
        
        # Initialize core models
        gpt4_model = GPT4Model(temperature=0.7)
        chess_transformer = ChessTransformer(
            encoder_layers=8,
            attention_heads=12,
            d_model=768
        )
        
        # Ensure books directory exists
        books_path = Path("data/books")
        if not books_path.exists():
            logger.warning(f"Opening books directory not found at {books_path}")
            books_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize hybrid architecture
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
"""

# Set up RAG system
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(chess_knowledge)
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])
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
ai_thinking = False
ai_interval = None

# Initialize AI game control
def stop_ai_game():
    """Stop AI game and clean up resources"""
    global ai_interval, game_over
    if ai_interval:
        ai_interval = None
    game_over = True
    emit('game_stopped', {'message': 'Game completed'})

class GameTracker:
    def __init__(self, evaluator):
        self.moves = []
        self.captures = []
        self.checks = []
        self.castlings = []
        self.material_balance = []
        self.position_scores = []
        self.time_per_move = []
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    def add_move(self, board: chess.Board, move: chess.Move, time_taken: float = 0.0):
        """Track a new move and its implications"""
        try:
            # Record basic move info
            san_move = board.san(move)
            self.moves.append(san_move)
            self.time_per_move.append(time_taken)
            
            # Track special moves
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                self.captures.append((san_move, captured_piece))
                self.logger.info(f"Capture recorded: {san_move} takes {captured_piece}")
            
            if board.is_check():
                self.checks.append(san_move)
                self.logger.info(f"Check recorded: {san_move}")
            
            if board.is_castling(move):
                self.castlings.append(san_move)
                self.logger.info(f"Castling recorded: {san_move}")
            
            # Get position evaluation using evaluator
            try:
                board_copy = board.copy()
                board_copy.push(move)
                evaluation = self.evaluator.evaluate_position(board_copy)
                self.position_scores.append(evaluation)
            except Exception as e:
                self.logger.error(f"Error in evaluation: {str(e)}")
                self.position_scores.append(0.0)
            
            self.logger.info(f"Move {san_move} fully processed and tracked")
            
        except Exception as e:
            self.logger.error(f"Error tracking move: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        """Get comprehensive game statistics"""
        return {
            'total_moves': len(self.moves),
            'captures': len(self.captures),
            'checks': len(self.checks),
            'castlings': len(self.castlings),
            'average_time': sum(self.time_per_move) / len(self.time_per_move) if self.time_per_move else 0,
            'material_trajectory': self.material_balance,
            'position_scores': self.position_scores
        }

# Initialize game tracker with evaluator
game_tracker = GameTracker(evaluator=hybrid_brain)

def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str, float]:
    """Get the best move using the hybrid AI system"""
    try:
        board = chess.Board(board_fen)
        move, explanation = hybrid_brain.get_move(board, legal_moves)
        
        # Make the move on a copy of the board to evaluate position
        board_copy = board.copy()
        try:
            chess_move = chess.Move.from_uci(move)
            board_copy.push(chess_move)
            evaluation = hybrid_brain.evaluate_position(board_copy)
        except Exception as e:
            logger.error(f"Error in position evaluation: {str(e)}")
            evaluation = 0.0
        
        logger.info(f"Move: {move}, Evaluation: {evaluation}")
        return move, explanation, evaluation
        
    except Exception as e:
        logger.error(f"Error in get_best_move: {str(e)}")
        return legal_moves[0], "Fallback move selected", 0.0

def get_legal_moves() -> str:
    """Get all legal moves in current position"""
    return "Possible moves are: " + ",".join([move.uci() for move in board.legal_moves])

def is_game_over() -> bool:
    """Enhanced game over check"""
    global game_over
    
    if board.is_checkmate():
        game_over = True
        return True
        
    if board.is_stalemate():
        game_over = True
        return True
        
    if board.is_insufficient_material():
        game_over = True
        return True
        
    if board.is_fifty_moves():
        game_over = True
        return True
        
    if board.is_repetition(3):
        game_over = True
        return True
        
    if move_count >= 100:  # 50 moves per side
        game_over = True
        return True
        
    return False

def check_made_move(msg: str) -> bool:
    """Check if a move was made and update game status"""
    global made_move, game_over
    if made_move:
        made_move = False
        game_over = is_game_over()
    return game_over

def make_move(move: str, explanation: str = "") -> Tuple[str, str, bool]:
    """Make a move on the board with enhanced tracking"""
    global made_move, board, move_count, game_over, game_tracker
    
    if game_over:
        return "The game is already over.", explanation, True

    try:
        start_time = time.time()
        chess_move = chess.Move.from_uci(move)
        
        if chess_move in board.legal_moves:
            # Track move before making it
            game_tracker.add_move(board, chess_move, time.time() - start_time)
            
            # Make the move
            board.push(chess_move)
            made_move = True
            move_count += 1

            # Generate move description
            piece = board.piece_at(chess_move.to_square)
            piece_symbol = piece.symbol() if piece else ''
            piece_name = chess.piece_name(piece.piece_type) if piece else ''

            result = f"Moved {piece_name} ({piece_symbol}) from "\
                     f"{chess.SQUARE_NAMES[chess_move.from_square]} to "\
                     f"{chess.SQUARE_NAMES[chess_move.to_square]}"

            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                captured_piece_name = chess.piece_name(captured_piece.piece_type)
                result += f". Captured {captured_piece_name}."

            # Check game ending conditions
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

def summarize_game(tracker: GameTracker, result: str, total_moves: int) -> str:
    """Generate comprehensive game summary"""
    try:
        summary_prompt = f"""
        Analyze this chess game concisely:
        Game Result: {result}
        Total Moves: {total_moves}
        Opening Moves: {', '.join(tracker.moves[:10])}
        Key Statistics:
        - Captures: {len(tracker.captures)}
        - Checks: {len(tracker.checks)}
        - Castling moves: {', '.join(tracker.castlings)}
        - Position Scores: {tracker.position_scores[-5:] if tracker.position_scores else 'No scores recorded'}
        
        Recent Activity:
        - Last few moves: {', '.join(tracker.moves[-5:])}
        - Recent captures: {', '.join([f"{move} ({piece})" for move, piece in tracker.captures[-3:]])}
        - Recent checks: {', '.join(tracker.checks[-3:])}
        
        Provide a brief, clear summary focusing on:
        1. The opening and its effectiveness
        2. Key tactical moments and strategic themes
        3. Critical turning points
        4. How and why the game concluded as it did
        """

        summary = qa.run(summary_prompt)
        logger.info("Game summary generated successfully")
        return summary
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    emit('game_state', {
        'fen': board.fen(),
        'legal_moves': [move.uci() for move in board.legal_moves]
    })
    logger.info("Client connected")

@socketio.on('request_ai_move')
def handle_ai_move():
    """Handle AI move requests with proper game termination"""
    global ai_thinking
    try:
        # Check if game is over or AI is thinking
        if game_over:
            emit('game_stopped', {
                'message': 'Game is already over',
                'summary': summarize_game(game_tracker, board.result(), move_count)
            })
            return
            
        if ai_thinking:
            return

        ai_thinking = True
        logger.info("AI move requested")
        
        best_move, explanation, evaluation = get_best_move(
            board.fen(), 
            [m.uci() for m in board.legal_moves]
        )
        
        result, explanation, is_game_over = make_move(best_move, explanation)
        
        emit('move_made', {
            'move': best_move,
            'result': result,
            'explanation': explanation,
            'evaluation': evaluation,
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves],
            'game_over': is_game_over
        })
        
        if is_game_over:
            summary = summarize_game(game_tracker, board.result(), move_count)
            emit('game_summary', {'summary': summary})
            stop_ai_game()
            
    except Exception as e:
        logger.error(f"Error in AI move: {str(e)}")
        emit('error', {'message': str(e)})
    finally:
        ai_thinking = False

@socketio.on('make_move')
def handle_make_move(data):
    """Handle human move requests"""
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
        
        if is_game_over:
            summary = summarize_game(game_tracker, board.result(), move_count)
            emit('game_summary', {'summary': summary})
            
    except Exception as e:
        logger.error(f"Error handling move: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('reset_game')
def handle_reset_game():
    """Handle game reset with proper initialization"""
    try:
        global board, move_count, game_over, game_tracker, ai_thinking, ai_interval
        
        # Reset game state
        board = chess.Board()
        move_count = 0
        game_over = False
        ai_thinking = False
        
        # Clear AI interval if exists
        if ai_interval:
            ai_interval = None
        
        # Initialize new game tracker
        game_tracker = GameTracker(evaluator=hybrid_brain)
        
        # Send new game state
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
    """Handle game summary requests"""
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
        
        # Stop AI game
        stop_ai_game()
        
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
    """Handle PGN export requests"""
    try:
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "AI Chess Game"
        game.headers["Site"] = "Python Flask App"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "AI"
        game.headers["Black"] = "AI"
        game.headers["Result"] = board.result()

        pgn_string = StringIO()
        exporter = chess.pgn.FileExporter(pgn_string)
        game.accept(exporter)
        
        emit('pgn_data', {'pgn': pgn_string.getvalue()})
        logger.info("PGN data generated and sent")
    except Exception as e:
        logger.error(f"Error getting PGN: {str(e)}")
        emit('error', {'message': str(e)})

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/export_pgn')
def export_pgn():
    try:
        pgn = export_game_to_pgn(board)
        return pgn, 200, {
            'Content-Type': 'text/plain',
            'Content-Disposition': 'attachment; filename=chess_game.pgn'
        }
    except Exception as e:
        logger.error(f"Error exporting PGN: {str(e)}")
        return str(e), 500

# Error Handlers
@socketio.on_error()
def handle_error(e):
    """Global WebSocket error handler"""
    logger.error(f"WebSocket error: {str(e)}")
    emit('error', {'message': 'An error occurred', 'details': str(e)})

@app.errorhandler(Exception)
def handle_exception(e):
    """Global HTTP error handler"""
    logger.error(f"HTTP error: {str(e)}")
    return {'success': False, 'error': str(e)}, 500

@app.after_request
def after_request(response):
    cleanup_memory()
    return response

# Application Startup
if __name__ == '__main__':
    try:
        port = 5001
        host = '0.0.0.0'
        logger.info("Starting the Chess AI application")
        logger.info(f"Server will be available at http://localhost:{port}")
        
        # Log startup status
        log_startup_status()
        
        # Run the server
        socketio.run(app, debug=False, host=host, port=port, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise