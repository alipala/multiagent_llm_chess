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
from models.base_model import ChessEngine
from typing import Any

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
def setup_logging():
    """Configure logging with better formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress noisy loggers
    noisy_loggers = [
        'engineio.server',
        'socketio.server',
        'chromadb',
        'httpx',
        'autogen'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Setup logging early
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__, static_url_path='/static')
app.config['SERVER_NAME'] = None 
# It is for local run // socketio = SocketIO(app, cors_allowed_origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)


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
        
        # Initialize Chess Engine with correct ChatOpenAI integration        
        chess_engine = ChessEngine(temperature=0.7)
    
        logger.info("AI models initialized successfully")
        return chess_engine
    except Exception as e:
        logger.error(f"Error initializing AI models: {str(e)}")
        raise

# Initialize engines
try:
    chess_engine = initialize_models() 
    logger.info("Chess engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chess engine: {str(e)}")
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
# Opening Principles
1. Essential Development Rules:
   - Don't move the same piece twice in the opening
   - Develop knights before bishops
   - Don't bring queen out too early
   - Castle within the first 7-8 moves
   - Control center with pawns (e4, d4, e5, d5)
   - Only make pawn moves that aid development

2. Common Opening Mistakes to Avoid:
   - Moving edge pawns (a,h) too early
   - Making too many pawn moves
   - Moving queen prematurely
   - Making pointless knight moves
   - Weakening king's position

3. Center Control Strategy:
   - Occupy center with pawns first (e4/d4 or e5/d5)
   - Support center pawns with minor pieces
   - Don't exchange center pawns without clear benefit
   - Maintain tension when advantageous

# Middlegame Strategy
1. King Safety Priority:
   - Complete castling before attacking
   - Maintain pawn shield in front of castled king
   - Watch for diagonal weaknesses
   - Don't advance pawns in front of castled king without purpose

2. Piece Coordination:
   - Connect rooks after castling
   - Place bishops on active diagonals
   - Establish knights on strong outposts
   - Create piece chains protecting each other
   - Coordinate pieces before launching attacks

3. Attack Prerequisites:
   - Ensure king safety first
   - Have more pieces in attacking zone
   - Control key squares around enemy king
   - Create weaknesses in enemy position
   - Don't attack without proper preparation

# Position Evaluation
1. Material Balance:
   - Consider piece values (P=1, N=3, B=3, R=5, Q=9)
   - Bishop pair is worth extra half-pawn
   - Knights strong in closed positions
   - Bishops strong in open positions

2. Positional Factors:
   - Pawn structure health
   - Piece activity and coordination
   - King safety assessment
   - Control of key squares and files
   - Development lead
   - Space advantage

3. Dynamic Elements:
   - Piece mobility
   - Attacking chances
   - Tactical opportunities
   - Pawn breaks
   - Piece coordination potential

# Common Tactical Patterns
1. Basic Tactics:
   - Fork: One piece attacks two
   - Pin: Piece can't move due to exposure
   - Skewer: Similar to pin but higher value piece in front
   - Discovery: Moving one piece reveals attack from another

2. Tactical Motifs:
   - Overloading: Piece defending too many squares
   - Deflection: Forcing piece away from defense
   - Clearance: Removing blocking piece
   - Interference: Blocking defensive piece

# Safety Checks Before Moving
1. Pre-Move Checklist:
   - Check all opponent's captures
   - Look for tactical threats
   - Consider opponent's best reply
   - Evaluate resulting position
   - Verify move aids overall plan

2. Position Maintenance:
   - Keep pieces protected
   - Maintain pawn structure
   - Watch diagonal weaknesses
   - Control key squares
   - Keep king safe
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
        """Track move after it has been made"""
        try:
            # Get move in UCI format
            move_uci = move.uci()
            self.moves.append(move_uci)
            self.time_per_move.append(time_taken)
            
            try:
                # Track position evaluation using the evaluator
                evaluation = self.evaluator.evaluate_position(board)
                self.position_scores.append(evaluation)
                
                # Track material balance
                material = self.evaluator._evaluate_material(board)
                self.material_balance.append(material)
            except Exception as e:
                self.logger.warning(f"Error in evaluation tracking: {str(e)}")
                self.position_scores.append(0.0)
                self.material_balance.append(0.0)
            
            # Track special moves
            if board.is_capture(move):
                captured_sq = move.to_square
                captured_piece = board.piece_at(captured_sq)
                if captured_piece:
                    self.captures.append((move_uci, captured_piece))
                    
            if board.is_check():
                self.checks.append(move_uci)
                
            if board.is_castling(move):
                self.castlings.append(move_uci)
                
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

    def evaluate_position(self, board: chess.Board) -> float:
        """Enhanced position evaluation"""
        try:
            material_score = self._evaluate_material(board)
            positional_score = self._evaluate_positional_factors(board)
            mobility_score = self._evaluate_mobility(board)
            king_safety_score = self._evaluate_king_safety(board)
            pawn_structure_score = self._evaluate_pawn_structure(board)
            
            # Weighted combination
            total_score = (
                0.60 * material_score +
                0.15 * positional_score +
                0.10 * mobility_score +
                0.10 * king_safety_score +
                0.05 * pawn_structure_score
            ) / 100.0  # Convert centipawns to pawns
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return 0.0

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

game_tracker = GameTracker(chess_engine)

# Game functions
def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str, float]:
    """Get best move with single LLM call"""
    try:
        board = chess.Board(board_fen)
        player = 'White' if board.turn else 'Black'
        logger.info(f"🎮 {player} is thinking...")
        
        # Get move and explanation from chess engine
        move, explanation = chess_engine.get_move(board, legal_moves)
        
        # Quick evaluation without API call
        evaluation = chess_engine.evaluate_position(board)
        
        logger.info(f"🎯 {player} plays {move}")
        return move, explanation, evaluation
        
    except Exception as e:
        logger.error(f"⚠️ Error in move generation: {str(e)}")
        # Return first legal move instead of falling back
        return legal_moves[0], "Fallback move selected", 0.0
    
def analyze_position(board: chess.Board) -> str:
    """Analyze current position to create RAG query"""
    position_details = []
    
    # Game phase
    move_number = board.fullmove_number
    if move_number <= 10:
        position_details.append("opening phase")
    elif move_number <= 30:
        position_details.append("middlegame phase")
    else:
        position_details.append("endgame phase")
    
    # Material count
    material = {
        'P': len(board.pieces(chess.PAWN, chess.WHITE)),
        'p': len(board.pieces(chess.PAWN, chess.BLACK)),
        'N': len(board.pieces(chess.KNIGHT, chess.WHITE)),
        'n': len(board.pieces(chess.KNIGHT, chess.BLACK)),
        'B': len(board.pieces(chess.BISHOP, chess.WHITE)),
        'b': len(board.pieces(chess.BISHOP, chess.BLACK)),
        'R': len(board.pieces(chess.ROOK, chess.WHITE)),
        'r': len(board.pieces(chess.ROOK, chess.BLACK)),
        'Q': len(board.pieces(chess.QUEEN, chess.WHITE)),
        'q': len(board.pieces(chess.QUEEN, chess.BLACK))
    }
    
    # Add relevant position characteristics
    if board.is_check():
        position_details.append("king is in check")
    if len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) == 0:
        position_details.append("queens are exchanged")
    if len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK)) < 8:
        position_details.append("open pawn structure")
        
    return f"Position analysis: {', '.join(position_details)}. What strategic principles apply?"

def get_legal_moves() -> str:
    """Get all legal moves in current position"""
    return "Possible moves are: " + ",".join([move.uci() for move in board.legal_moves])

def make_move(move: str, explanation: str = "") -> Tuple[str, str, bool]:
    """Make a move on the board and return result"""
    global made_move, board, move_count, game_over, game_tracker
    
    if game_over:
        return "The game is already over.", explanation, True

    try:
        # Parse move from UCI format
        from_square = chess.parse_square(move[0:2])
        to_square = chess.parse_square(move[2:4])
        promotion = chess.QUEEN if len(move) > 4 else None
        chess_move = chess.Move(from_square, to_square, promotion=promotion)
        
        if chess_move not in board.legal_moves:
            logger.warning(f"⚠️ Illegal move attempted: {move}")
            return f"Illegal move: {move}. Legal moves are: {get_legal_moves()}", explanation, game_over

        # Get piece info before the move
        moving_piece = board.piece_at(from_square)
        piece_name = chess.piece_name(moving_piece.piece_type) if moving_piece else ''
        from_square_name = chess.square_name(from_square)
        to_square_name = chess.square_name(to_square)
        
        # Record special moves before making the move
        is_capture = board.is_capture(chess_move)
        captured_piece = board.piece_at(to_square) if is_capture else None
        
        # Track timing
        start_time = time.time()
        
        # Make the move on the board
        board.push(chess_move)
        
        # Update tracking after move is made
        game_tracker.add_move(board, chess_move, time.time() - start_time)
        
        # Build result message
        result = [f"Moved {piece_name} from {from_square_name} to {to_square_name}"]
        if is_capture and captured_piece:
            result.append(f"Captured {chess.piece_name(captured_piece.piece_type)}")
        if board.is_check():
            result.append("Check!")
            
        # Update game state
        made_move = True
        move_count += 1
        game_over = is_game_over()
        
        # Finalize result message
        result_str = ". ".join(result)
        if game_over:
            result_str += _get_game_over_message()
            
        logger.info(f"🎯 Move made: {result_str}")
        if explanation:
            logger.info(f"💭 Reasoning: {explanation}")
            
        return result_str, explanation, game_over
        
    except ValueError as e:
        logger.error(f"⚠️ Invalid move format: {move}, Error: {str(e)}")
        return f"Invalid move format: {move}. Please use UCI format (e.g., 'e2e4').", explanation, game_over


def _execute_move(chess_move: chess.Move) -> str:
    """Execute the move and generate description"""
    # Get piece info
    moving_piece = board.piece_at(chess_move.from_square)
    piece_name = chess.piece_name(moving_piece.piece_type) if moving_piece else ''
    piece_symbol = moving_piece.symbol() if moving_piece else ''
    
    # Get square names
    from_square = chess.SQUARE_NAMES[chess_move.from_square]
    to_square = chess.SQUARE_NAMES[chess_move.to_square]
    
    # Build result message
    result = [f"Moved {piece_name} ({piece_symbol}) from {from_square} to {to_square}"]
    
    # Check for capture
    if board.is_capture(chess_move):
        captured_piece = board.piece_at(chess_move.to_square)
        if captured_piece:
            result.append(f"Captured {chess.piece_name(captured_piece.piece_type)}")
    
    # Make the move
    board.push(chess_move)
    
    # Add check indication
    if board.is_check():
        result.append("Check!")
        
    return ". ".join(result) + ("" if result[-1].endswith("!") else ".")

def _get_game_over_message() -> str:
    """Generate game over message"""
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
    else:
        return " The game is over."
        
def generate_commentary(board: Any, move: str, position_eval: float) -> str:
    logger.info(f"Commentator_Agent: Generating commentary for move {move}")
    try:
        chess_move = chess.Move.from_uci(move)
        piece = board.piece_at(chess_move.from_square)
        prompt = f"""Comment on this chess position:
        - {chess.piece_name(piece.piece_type)} to {chess.square_name(chess_move.to_square)}
        - Capture: {board.is_capture(chess_move)}
        - Check: {board.gives_check(chess_move)}
        - Current evaluation: {position_eval:.2f}
        - Move number: {board.fullmove_number}"""
        
        return commentator_agent.generate(prompt).response
    except Exception as e:
        logger.error(f"Commentary error: {str(e)}")
        return "A strategic move in this position."

def is_game_over() -> bool:
    """Enhanced game over check"""
    global game_over
    
    # Regular chess endings
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        game_over = True
        return True
        
    # Repetition draw
    if board.is_repetition(3) or board.is_fifty_moves():
        game_over = True
        return True
        
    # Move limit (optional, can be adjusted or removed)
    if move_count >= 40:  # Increased from 20 to 40
        game_over = True
        return True
        
    # Mutual attacks check using correct python-chess methods
    if board.turn == chess.WHITE:
        white_king_square = board.king(chess.WHITE)
        if white_king_square is not None and board.attackers_mask(chess.BLACK, white_king_square):
            game_over = True
            return True
    else:
        black_king_square = board.king(chess.BLACK)
        if black_king_square is not None and board.attackers_mask(chess.WHITE, black_king_square):
            game_over = True
            return True
        
    return False

def handle_game_end():
    """Enhanced game end handling with better logging"""
    try:
        if not game_over:
            logger.info("handle_game_end called but game is not over")
            return
            
        logger.info("=== Handling Game End ===")
        logger.info(f"Final Position: {board.fen()}")
        logger.info(f"Game Result: {board.result() or '*'}")
        
        # Generate game summary with retries
        max_retries = 3
        summary = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to generate summary (attempt {attempt + 1}/{max_retries})")
                summary = summarize_game(game_tracker, board.result(), move_count)
                if summary:
                    logger.info("Summary generated successfully!")
                    logger.info("Summary content:")
                    logger.info("-" * 50)
                    logger.info(summary)
                    logger.info("-" * 50)
                    break
            except Exception as e:
                logger.warning(f"Summary generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(1)
        
        if not summary:
            summary = "A summary could not be generated at this time."
            logger.warning("Failed to generate summary after all attempts")
        
        # Get captured pieces
        white_captured = []
        black_captured = []
        try:
            for capture in game_tracker.captures:
                if isinstance(capture, tuple) and len(capture) == 2 and capture[1]:
                    piece_str = chess.piece_name(capture[1].piece_type)
                    if capture[1].color == chess.WHITE:
                        white_captured.append(piece_str)
                    else:
                        black_captured.append(piece_str)
        except Exception as e:
            logger.error(f"Error processing captures: {str(e)}")
        
        # Prepare game over data
        game_over_data = {
            'result': board.result() or '*',
            'summary': summary,
            'final_position': board.fen(),
            'move_count': move_count,
            'white_captured': white_captured,
            'black_captured': black_captured,
            'status': _get_game_over_message()
        }
        
        # Log what we're about to emit
        logger.info("Preparing to emit game_over event with data:")
        logger.info(f"Result: {game_over_data['result']}")
        logger.info(f"Move Count: {game_over_data['move_count']}")
        logger.info(f"White Captured: {game_over_data['white_captured']}")
        logger.info(f"Black Captured: {game_over_data['black_captured']}")
        logger.info(f"Status: {game_over_data['status']}")
        
        # Emit game over event
        try:
            socketio.emit('game_over', game_over_data)
            logger.info("Successfully emitted game_over event")
        except Exception as e:
            logger.error(f"Error emitting game over event: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in handle_game_end: {str(e)}", exc_info=True)
        socketio.emit('error', {
            'message': 'Error handling game end',
            'details': str(e)
        })

def check_made_move(msg: str) -> bool:
    """Check if a move was made and update game status"""
    global made_move, game_over
    if made_move:
        made_move = False
        game_over = is_game_over()
    return game_over

def summarize_game(tracker: GameTracker, result: str, total_moves: int) -> str:
    """Enhanced game summary generation with detailed logging"""
    try:
        logger.info("=== Starting Game Summary Generation ===")
        
        # Get game information
        opening_phase = tracker.moves[:10]
        material_changes = tracker.material_balance
        position_changes = tracker.position_scores
        game_phase = _determine_game_phase(total_moves)
        
        # Log the input data
        logger.info(f"Game Data for Summary:")
        logger.info(f"Result: {result}")
        logger.info(f"Total Moves: {total_moves}")
        logger.info(f"Opening Moves: {', '.join(opening_phase[:5])}")
        logger.info(f"Material Balance: Start={material_changes[0] if material_changes else 0}, End={material_changes[-1] if material_changes else 0}")
        logger.info(f"Captures: {len(tracker.captures)}")
        
        summary_prompt = f"""
        Chess Game Analysis Summary:
        
        Result: {result}
        Total Moves: {total_moves}
        Opening Phase: {', '.join(opening_phase[:5])}
        Material Changes: Initial {material_changes[0] if material_changes else 0} → Final {material_changes[-1] if material_changes else 0}
        Position Score Changes: Initial {position_changes[0] if position_changes else 0} → Final {position_changes[-1] if position_changes else 0}
        Game Phase Reached: {game_phase}
        Key Events:
        - Captures: {len(tracker.captures)}
        - Checks: {len(tracker.checks)}
        - Castling Moves: {len(tracker.castlings)}
        
        Please summarize this chess game concisely, focusing on:
        1. Key turning points
        2. Decisive moments
        3. Final position evaluation
        4. Winning factors

        Keep summary brief but informative.
        """
        
        logger.info("Generating summary using LangChain...")
        
        # Use new invoke method with error handling
        try:
            response = qa.invoke({"query": summary_prompt})
            summary = response.get('result', '')
            logger.info("Raw LangChain Response:")
            logger.info(response)
        except Exception as e:
            logger.error(f"LangChain invoke error: {str(e)}")
            summary = "Error generating detailed summary."
        
        if not summary:
            summary = "The game concluded with the given result after a series of tactical exchanges."
            logger.warning("Empty summary generated, using default message")
        else:
            logger.info("Generated Summary:")
            logger.info("=" * 50)
            logger.info(summary)
            logger.info("=" * 50)
            
        return summary
        
    except Exception as e:
        logger.error(f"Error in summarize_game: {str(e)}", exc_info=True)
        return "A game summary could not be generated at this time."

def _determine_game_phase(total_moves: int) -> str:
    if total_moves <= 10:
        return "Opening"
    elif total_moves <= 30:
        return "Middlegame"
    else:
        return "Endgame"
    
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

# System message for White player
white_player_system_message = """
You are a chess player who combines aggressive style with sound principles. Your decision making process:

1. Opening Phase (moves 1-10):
   - Start with e4 or d4 only
   - Develop knights to f3 or c3
   - Don't move queen before move 7
   - Castle within first 7 moves
   - Don't attack unless fully developed
   - Control center squares (e4,d4,e5,d5)

2. Safety Checklist (EVERY move):
   - Is my king safe?
   - Are all my pieces protected?
   - What captures can opponent make?
   - Does this move help development?
   - Is there a better square for this piece?

3. Aggressive Strategy:
   - Only attack AFTER:
     * Completed development
     * Castled for king safety
     * Secured center control
     * Connected rooks
   - Look for:
     * Tactical opportunities
     * Piece sacrifices with clear compensation
     * Pawn breaks to open position
     * Direct attacks against king

4. Move Selection Process:
   1. Check all captures
   2. Check all checks
   3. Look for tactical shots
   4. Consider positional improvements
   5. Choose move that follows opening principles
   6. Verify move is safe

Before EACH move:
1. Use get_legal_moves() to see options
2. Use get_best_move() for evaluation
3. Verify move follows above principles
4. Explain your reasoning
"""

# System message for Black player
black_player_system_message = """
You are a chess player who focuses on solid positional play. Your decision making process:

1. Opening Phase (moves 1-10):
   - Respond to e4 with e5 or c5
   - Respond to d4 with d5 or nf6
   - Develop knights first
   - Don't move queen before move 7
   - Castle within first 8 moves
   - Control center squares

2. Safety Checklist (EVERY move):
   - Is my king safe?
   - Are all my pieces protected?
   - What captures can opponent make?
   - Does this move help development?
   - Is there a better square for this piece?

3. Positional Strategy:
   - Focus on:
     * Solid pawn structure
     * Piece coordination
     * Control of key squares
     * Bishop pair advantage
     * Knight outposts
   - Avoid:
     * Pawn weaknesses
     * Isolated pawns
     * Exposed king
     * Undefended pieces

4. Move Selection Process:
   1. Check all opponent threats
   2. Look for defensive resources
   3. Find pawn structure improvements
   4. Position pieces actively
   5. Create long-term advantages
   6. Verify move safety

Before EACH move:
1. Use get_legal_moves() to see options
2. Use get_best_move() for evaluation
3. Verify move follows above principles
4. Explain your reasoning
"""

# Create player agents
player_white = ConversableAgent(
    name="Player_White",
    system_message=white_player_system_message,
    llm_config=llm_config,
)

player_black = ConversableAgent(
    name="Player_Black",
    system_message=black_player_system_message,
    llm_config=llm_config,
)

board_proxy = ConversableAgent(
    name="Board_Proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

commentator_agent = ConversableAgent(
    name="Chess_Commentator",
    system_message="""You are an enthusiastic chess commentator providing engaging, insightful commentary.
    Focus on:
    1. Tactical elements (captures, threats, combinations)
    2. Strategic implications (pawn structure, piece placement)
    3. Position evaluation and potential plans
    4. Historical context of similar positions""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Function registration
for caller in [player_white, player_black]:
    register_function(is_game_over, caller=caller, executor=board_proxy, name="is_game_over", description="Check if the game is over.")
    register_function(make_move, caller=caller, executor=board_proxy, name="make_move", description="Make a move on the chess board.")
    register_function(get_best_move, caller=caller, executor=board_proxy, name="get_best_move", description="Get the best move based on Chess engine analysis.")
    register_function(get_legal_moves, caller=caller, executor=board_proxy, name="get_legal_moves", description="Get a list of legal moves in the current position.")
    register_function(generate_commentary, caller=commentator_agent, executor=board_proxy, name="generate_commentary", description="Generate chess commentary")

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

for player in [player_white, player_black]:
    player.register_nested_chats(
        trigger=commentator_agent, 
        chat_queue=[{"sender": commentator_agent, "recipient": player}]
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

@socketio.on('connect', namespace='/')
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
        
        move_data = {
            'move': move,
            'result': result,
            'explanation': explanation,
            'fen': board.fen(),
            'legal_moves': [m.uci() for m in board.legal_moves],
            'game_over': is_game_over
        }
        
        emit('move_made', move_data)
        
        if is_game_over:
            handle_game_end()
            
        logger.info(f"Move handled: {move}")
    except Exception as e:
        logger.error(f"Error handling move: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('request_ai_move')
def handle_ai_move():
    try:
        logger.info("AI move requested")
        
        if game_over:
            handle_game_end()
            return
            
        # Rest of the AI move handling code...
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
            handle_game_end()
            
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
        game_tracker = GameTracker(evaluator=chess_engine)
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
    """Handle game summary request with improved logging"""
    try:
        logger.info("=== Handling Get Summary Request ===")
        
        if not game_over:
            logger.warning("Summary requested but game is not over")
            emit('game_summary', {
                'error': 'Game is not over yet',
                'summary': None
            })
            return
            
        logger.info("Generating summary...")
        summary = summarize_game(game_tracker, board.result(), move_count)
        
        logger.info("Emitting summary:")
        logger.info(summary)
        
        emit('game_summary', {
            'summary': summary,
            'error': None
        })
        logger.info("Summary emitted successfully")
        
    except Exception as e:
        logger.error(f"Error in handle_get_summary: {str(e)}", exc_info=True)
        emit('game_summary', {
            'error': str(e),
            'summary': None
        })

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