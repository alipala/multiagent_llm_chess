import os
import chess
import chess.svg
import chess.pgn
from typing import List, Union
from typing_extensions import Annotated
import openai
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

# Suppress specific Autogen warnings
warnings.filterwarnings("ignore", message="Function .* is being overridden", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
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
Chess Opening Principles:
1. Control the center
2. Develop your pieces
3. Castle early
4. Don't move the same piece twice in the opening
5. Don't bring your queen out too early

Chess Middlegame Strategies:
1. Create and exploit weaknesses
2. Control open files and diagonals
3. Coordinate your pieces
4. Plan your pawn breaks
5. Look for tactical opportunities

Chess Endgame Techniques:
1. Activate your king
2. Create passed pawns
3. Cut off the opponent's king
4. Know basic checkmate patterns
5. Understand opposition and zugzwang

General Chess Tips:
1. Always have a plan
2. Calculate variations thoroughly
3. Watch for your opponent's threats
4. Improve your piece placement
5. Study classic games and puzzles
"""

# Set up RAG system
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(chess_knowledge)
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])

# Create a retriever with a limit on the number of results
retriever = docsearch.as_retriever(search_kwargs={"k": 1})

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

    def add_move(self, board: chess.Board, move: chess.Move):
        san_move = board.san(move)
        self.moves.append(san_move)
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            self.captures.append((san_move, captured_piece))
        if board.is_check():
            self.checks.append(san_move)
        if board.is_castling(move):
            self.castlings.append(san_move)

game_tracker = GameTracker()

# Game functions
def get_best_move(board_fen: str, legal_moves: List[str]) -> Tuple[str, str]:
    query = f"""Given the current chess position (FEN: {board_fen}) and legal moves {legal_moves}, 
    what's the best move? Respond with the move in UCI format followed by a brief explanation of why it's a good move. 
    Format your response as: 'move: explanation'"""
    result = qa.run(query)
    move, explanation = result.split(': ', 1)
    return move.strip(), explanation.strip()

def get_legal_moves() -> str:
    return "Possible moves are: " + ",".join([move.uci() for move in board.legal_moves])

def make_move(move: str, explanation: str = "") -> Tuple[str, str, bool]:
    global made_move, board, move_count, game_over, game_tracker
    if game_over:
        return "The game is already over.", explanation, True

    try:
        chess_move = chess.Move.from_uci(move)
        if chess_move in board.legal_moves:
            game_tracker.add_move(board, chess_move)
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
    global game_over
    game_over = game_over or board.is_game_over() or move_count >= 100  # 50 moves per player
    return game_over

def check_made_move(msg: str) -> bool:
    global made_move, game_over
    if made_move:
        made_move = False
        game_over = is_game_over()
    return game_over

def summarize_game(tracker: GameTracker, result: str, total_moves: int) -> str:
    summary_prompt = f"""
    As a chess expert, provide a brief summary of the following game:

    Total moves: {total_moves}
    Result: {result}
    Opening moves: {', '.join(tracker.moves[:5])}
    Total captures: {len(tracker.captures)}
    Total checks: {len(tracker.checks)}
    Castling moves: {', '.join(tracker.castlings)}

    Key events:
    - Captures: {', '.join([f"{move} (captured {piece})" for move, piece in tracker.captures[:5]])}
    - Checks: {', '.join(tracker.checks[:5])}

    Please provide a concise summary in about 3-5 sentences, focusing on:
    1. The opening played and its implications
    2. Key tactical or strategic moments, including significant captures or checks
    3. The game's progression and how it concluded (or the current state if unfinished)
    4. Any notable patterns or strategies employed by either player

    Base your analysis on sound chess principles and the information provided.
    """

    try:
        summary = qa.run(summary_prompt)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def export_game_to_pgn(board, white_name="Player_White", black_name="Player_Black"):
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

    return pgn_string.getvalue()

# Player setup
player_system_message = """
You are a chess player with deep knowledge of chess principles.
Before making a move, check if the game is over by calling is_game_over().
If the game is not over:
1. Use the get_legal_moves function to get the list of legal moves.
2. Use the get_best_move function to determine the best move.
3. Call make_move(move) to make the move. Use only the move in UCI format.
4. After a move is made, explain your reasoning based on chess principles.
5. End your message with 'Your move.' to prompt the other player.
If the game is over, respond with 'The game has ended.' and do not send any further messages.
"""

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
    register_function(make_move, caller=caller, executor=board_proxy, name="make_move", description="Call this tool to make a move.")
    register_function(get_best_move, caller=caller, executor=board_proxy, name="get_best_move", description="Get the best move based on chess principles.")
    register_function(get_legal_moves, caller=caller, executor=board_proxy, name="get_legal_moves", description="Get a list of legal moves in the current position.")

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
    pgn = export_game_to_pgn(board)
    return pgn, 200, {'Content-Type': 'text/plain', 'Content-Disposition': 'attachment; filename=chess_game.pgn'}


@socketio.on('connect')
def handle_connect():
    emit('game_state', {'fen': board.fen(), 'legal_moves': [move.uci() for move in board.legal_moves]})

@socketio.on('make_move')
def handle_make_move(data):
    move = data['move']
    result = make_move(move)
    emit('move_made', {
        'move': move,
        'result': result,
        'fen': board.fen(),
        'legal_moves': [m.uci() for m in board.legal_moves],
        'game_over': is_game_over()
    })

@socketio.on('request_ai_move')
def handle_ai_move():
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

@socketio.on('reset_game')
def handle_reset_game():
    global board, move_count, game_over, game_tracker
    board = chess.Board()
    move_count = 0
    game_over = False
    game_tracker = GameTracker()
    emit('game_state', {'fen': board.fen(), 'legal_moves': [move.uci() for move in board.legal_moves]})

@socketio.on('get_game_summary')
def handle_get_summary():
    summary = summarize_game(game_tracker, board.result(), move_count)
    emit('game_summary', {'summary': summary})

@socketio.on('get_pgn')
def handle_get_pgn():
    pgn = export_game_to_pgn(board)
    emit('pgn_data', {'pgn': pgn})

if __name__ == '__main__':
    logger.info("Starting the Chess AI application")
    socketio.run(app, debug=False, host='0.0.0.0', port=5001)