import os
import chess
import chess.pgn
from typing import List, Tuple, Optional
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from autogen import ConversableAgent, config_list_from_json
from dotenv import load_dotenv
from datetime import datetime
import time
from io import StringIO
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging
from models.base_model import GPT4Model, ChessTransformer, HybridArchitecture
import gc
from urllib.parse import urljoin
import platform
import psutil

from agents.integration import ChessAgentSystem
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("autogen").setLevel(logging.ERROR)

# Initialize Flask and SocketIO
app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Load environment variables and configure OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize AI models
def initialize_models():
    try:
        gpt4_model = GPT4Model(temperature=0.7)
        chess_transformer = ChessTransformer(
            encoder_layers=8,
            attention_heads=12,
            d_model=768
        )
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
hybrid_brain = initialize_models()

# Initialize agent system
chess_agent_system = ChessAgentSystem()

# Configure AutoGen
config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o"],
    },
)

# Initialize RAG components with initial data
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

try:
    # Initialize Chroma with initial knowledge
    docsearch = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=[{"source": f"chess_knowledge_{i}"} for i in range(len(texts))]
    )
    
    # Initialize retriever and QA chain
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o"),
        chain_type="stuff",
        retriever=retriever
    )
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG system: {e}")
    # Fallback to simpler QA without vector storage
    qa = None
    logger.warning("Running without RAG capabilities")

# Game state variables
board = chess.Board()
made_move = False
move_count = 0
game_over = False
ai_thinking = False
ai_interval = None

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

    def add_move(self, board: chess.Board, move: chess.Move, time_taken: float = 0.0):
        try:
            san_move = board.san(move)
            self.moves.append(san_move)
            self.time_per_move.append(time_taken)
            
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                self.captures.append((san_move, captured_piece))
            
            if board.is_check():
                self.checks.append(san_move)
            
            if board.is_castling(move):
                self.castlings.append(san_move)
            
            try:
                board_copy = board.copy()
                board_copy.push(move)
                evaluation = self.evaluator.evaluate_position(board_copy)
                self.position_scores.append(evaluation)
            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
                self.position_scores.append(0.0)
            
        except Exception as e:
            logger.error(f"Error tracking move: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        return {
            'total_moves': len(self.moves),
            'captures': len(self.captures),
            'checks': len(self.checks),
            'castlings': len(self.castlings),
            'average_time': sum(self.time_per_move) / len(self.time_per_move) if self.time_per_move else 0,
            'material_trajectory': self.material_balance,
            'position_scores': self.position_scores
        }

game_tracker = GameTracker(evaluator=hybrid_brain)

def log_startup_status(host: str, port: int):
    """Log detailed startup status"""
    logger.info("="*50)
    logger.info("Chess AI Application Startup Status")
    logger.info("="*50)
    
    # System Information
    logger.info("System Information:")
    logger.info(f"OS: {platform.system()} {platform.version()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    memory = psutil.virtual_memory()
    logger.info(f"Memory Available: {memory.available / (1024 * 1024 * 1024):.1f}GB")
    
    # Server Configuration
    logger.info("\nServer Configuration:")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    
    # Access URLs
    logger.info("\nAccess URLs:")
    logger.info(f"Local URL: http://localhost:{port}")
    logger.info(f"Network URL: http://{host}:{port}")
    
    # Available Endpoints
    logger.info("\nAvailable Endpoints:")
    logger.info("└── / (Main chess interface)")
    logger.info("└── /export_pgn (PGN export)")
    
    # Components Status
    logger.info("\nComponent Status:")
    logger.info("├── Web Server: ✓ Running")
    logger.info("├── WebSocket Server: ✓ Ready")
    logger.info("├── Chess Engine: ✓ Initialized")
    logger.info("├── AI Models: ✓ Loaded")
    logger.info("└── RAG System: ✓ Ready")
    
    logger.info("="*50)
    logger.info("Server is ready for connections!")
    logger.info("="*50)

def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str, float]:
    try:
        board = chess.Board(board_fen)
        move, explanation = hybrid_brain.get_move(board, legal_moves)
        
        board_copy = board.copy()
        try:
            chess_move = chess.Move.from_uci(move)
            board_copy.push(chess_move)
            evaluation = hybrid_brain.evaluate_position(board_copy)
        except Exception as e:
            logger.error(f"Error in position evaluation: {str(e)}")
            evaluation = 0.0
        
        return move, explanation, evaluation
        
    except Exception as e:
        logger.error(f"Error in get_best_move: {str(e)}")
        return legal_moves[0], "Fallback move selected", 0.0

def is_game_over() -> bool:
    global game_over
    
    if board.is_checkmate() or board.is_stalemate() or \
       board.is_insufficient_material() or board.is_fifty_moves() or \
       board.is_repetition(3) or move_count >= 100:
        game_over = True
        return True
    return False

def make_move(move: str, explanation: str = "") -> Tuple[str, str, bool]:
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

            result = f"Moved {chess.piece_name(board.piece_at(chess_move.to_square).piece_type) if board.piece_at(chess_move.to_square) else ''} from "\
                     f"{chess.SQUARE_NAMES[chess_move.from_square]} to "\
                     f"{chess.SQUARE_NAMES[chess_move.to_square]}"

            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                result += f". Captured {chess.piece_name(captured_piece.piece_type)}."

            game_over = is_game_over()
            if game_over:
                result += _get_game_over_message()

            return result, explanation, game_over
        else:
            return f"Illegal move: {move}", explanation, game_over
    except ValueError:
        return f"Invalid move format: {move}", explanation, game_over

def _get_game_over_message() -> str:
    if board.is_checkmate():
        return f" Checkmate! {'White' if board.turn == chess.BLACK else 'Black'} wins."
    elif board.is_stalemate():
        return " Stalemate! The game is a draw."
    elif board.is_insufficient_material():
        return " Draw due to insufficient material."
    elif board.is_seventyfive_moves():
        return " Draw due to seventy-five moves rule."
    elif board.is_fivefold_repetition():
        return " Draw due to fivefold repetition."
    return " The game is over."

def summarize_game(tracker: GameTracker, result: str, total_moves: int) -> str:
    """Generate comprehensive game summary"""
    try:
        summary_prompt = f"""
        Analyze this chess game:
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
        """

        if qa:
            # Use sync version of chain
            response = qa({"query": summary_prompt})
            return response['result']
        else:
            # Fallback summary without RAG
            return f"Game ended with {result} after {total_moves} moves. "\
                   f"Notable events: {len(tracker.captures)} captures, {len(tracker.checks)} checks."
                   
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def cleanup_memory():
    gc.collect()
    
# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    emit('game_state', {
        'fen': board.fen(),
        'legal_moves': [move.uci() for move in board.legal_moves]
    })

@socketio.on('request_ai_move')
def handle_ai_move():
    global ai_thinking, board, game_over
    if game_over or ai_thinking:
        return

    try:
        ai_thinking = True
        logger.info(f"AI move requested. Current position: {board.fen()}")
        
        # Use our agent system to get the move
        response = asyncio.run(chess_agent_system.handle_move_request(board.fen()))
        
        if response['status'] == 'success' and response['move']:
            # Make the move
            move = chess.Move.from_uci(response['move'])
            
            if move in board.legal_moves:
                # Track game state before move
                is_capture = board.is_capture(move)
                captures_piece = board.piece_at(move.to_square) if is_capture else None
                
                # Make the move
                board.push(move)
                
                # Generate result text
                result = f"{'White' if board.turn == chess.BLACK else 'Black'}: {move.uci()}"
                if is_capture:
                    result += f" (Captured {captures_piece})"
                
                # Check game state
                is_game_over = (
                    board.is_checkmate() or 
                    board.is_stalemate() or 
                    board.is_insufficient_material() or 
                    board.is_fifty_moves()
                )
                
                # Emit move result
                emit('move_made', {
                    'move': move.uci(),
                    'result': result,
                    'explanation': response['explanation'],
                    'evaluation': response['evaluation'],
                    'fen': board.fen(),
                    'legal_moves': [m.uci() for m in board.legal_moves],
                    'game_over': is_game_over
                })
                
                if is_game_over:
                    game_over = True
                    logger.info("Game over detected. Generating summary...")
                    summary = summarize_game(game_tracker, board.result(), move_count)
                    emit('game_summary', {'summary': summary})
            else:
                emit('error', {'message': f"Illegal move suggested: {move.uci()}"})
        else:
            emit('error', {'message': "Failed to generate move"})
            
    except Exception as e:
        logger.error(f"Error in AI move: {str(e)}")
        emit('error', {'message': str(e)})
    finally:
        ai_thinking = False

@socketio.on('make_move')
def handle_make_move(data):
    """Handle manual moves from the frontend"""
    try:
        move = data['move']
        logger.info(f"Manual move received: {move}")
        
        # Validate move format
        if not move or len(move) < 4:
            emit('error', {'message': f"Invalid move format: {move}"})
            return
            
        try:
            chess_move = chess.Move.from_uci(move)
        except ValueError:
            emit('error', {'message': f"Invalid UCI move format: {move}"})
            return
            
        # Check if move is legal
        if chess_move not in board.legal_moves:
            emit('error', {'message': f"Illegal move: {move}"})
            return
            
        # Make the move
        board.push(chess_move)
        
        # Generate result text
        result = f"{'White' if board.turn == chess.BLACK else 'Black'}: {move}"
        if board.is_capture(chess_move):
            result += " (Captured piece)"
            
        # Check game state
        is_game_over = (
            board.is_checkmate() or 
            board.is_stalemate() or 
            board.is_insufficient_material() or 
            board.is_fifty_moves()
        )
        
        # Update game state in agent system
        asyncio.run(chess_agent_system.make_move(move))
        
        # Emit move result
        emit('move_made', {
            'move': move,
            'result': result,
            'explanation': "Manual move executed",
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves],
            'game_over': is_game_over
        })
        
        if is_game_over:
            summary = summarize_game(game_tracker, board.result(), move_count)
            emit('game_summary', {'summary': summary})
            
    except Exception as e:
        logger.error(f"Error handling manual move: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('reset_game')
def handle_reset_game():
    try:
        global board, move_count, game_over, game_tracker, ai_thinking
        board = chess.Board()
        move_count = 0
        game_over = False
        ai_thinking = False
        game_tracker = GameTracker(evaluator=hybrid_brain)
        
        # Reset agent system state
        asyncio.run(chess_agent_system.make_move('reset'))
        
        emit('game_state', {
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves]
        })
        
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

@socketio.on('get_pgn')
def handle_get_pgn():
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
        game = chess.pgn.Game.from_board(board)
        return str(game), 200, {
            'Content-Type': 'text/plain',
            'Content-Disposition': 'attachment; filename=chess_game.pgn'
        }
    except Exception as e:
        logger.error(f"Error exporting PGN: {str(e)}")
        return str(e), 500

@app.after_request
def after_request(response):
    cleanup_memory()
    return response

if __name__ == '__main__':
    try:
        host = '0.0.0.0'
        port = 5001
        
        # Log startup information
        log_startup_status(host, port)
        
        # Run the server
        socketio.run(app, debug=False, host=host, port=port, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise