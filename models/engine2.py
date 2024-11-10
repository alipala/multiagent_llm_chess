from pathlib import Path
import chess.polyglot
import chess
import random
from typing import Tuple, List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ChessEngine:
    def __init__(self, books_path: str = "data/books/"):
        self.books_path = Path(books_path)
        
        self.opening_books = {
            "gm2600.bin": {"weight": 0.6, "reader": None},    
            "Elo2400.bin": {"weight": 0.3, "reader": None},   
            "Performance.bin": {"weight": 0.08, "reader": None}, 
            "Book.bin": {"weight": 0.02, "reader": None}      
        }
        
        self._init_books()
        self.OPENING_MOVES = 15  # First 15 moves use book
        
        # Updated piece values with more granular evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Position bonuses
        self.center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        self.extended_center = [chess.E3, chess.E6, chess.D3, chess.D6, 
                              chess.C3, chess.C4, chess.C5, chess.C6,
                              chess.F3, chess.F4, chess.F5, chess.F6]

    def _init_books(self):
        """Initialize polyglot book readers with error handling"""
        try:
            books_loaded = 0
            for book_name, book_data in self.opening_books.items():
                book_path = self.books_path / book_name
                if book_path.exists():
                    try:
                        self.opening_books[book_name]["reader"] = chess.polyglot.open_reader(str(book_path))
                        books_loaded += 1
                    except Exception as e:
                        logger.error(f"Failed to load book {book_name}: {e}")
            logger.info(f"Successfully loaded {books_loaded} opening books")
        except Exception as e:
            logger.error(f"Error in book initialization: {e}")

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move from opening books with improved selection"""
        if board.fullmove_number > self.OPENING_MOVES:
            return None

        # First try GM book for best moves
        gm_reader = self.opening_books["gm2600.bin"]["reader"]
        if gm_reader:
            try:
                gm_moves = []
                max_weight = 0
                for entry in gm_reader.find_all(board):
                    if entry.move in board.legal_moves:
                        if entry.weight > max_weight:
                            gm_moves = [entry.move]
                            max_weight = entry.weight
                        elif entry.weight == max_weight:
                            gm_moves.append(entry.move)
                
                if gm_moves:
                    return random.choice(gm_moves)  # Choose randomly among best GM moves
            except Exception as e:
                logger.warning(f"Error accessing GM book: {e}")

        # If no GM move found, combine recommendations from all books
        all_moves = {}
        total_weight = 0
        best_move = None
        highest_weight = -1

        for book_name, book_data in self.opening_books.items():
            reader = book_data["reader"]
            if reader:
                try:
                    for entry in reader.find_all(board):
                        if entry.move in board.legal_moves:
                            weight = entry.weight * book_data["weight"]
                            all_moves[entry.move] = all_moves.get(entry.move, 0) + weight
                            total_weight += weight
                            
                            if weight > highest_weight:
                                highest_weight = weight
                                best_move = entry.move
                except Exception:
                    continue

        if best_move and total_weight > 0:
            # Use best move if it's significantly better
            best_weight = all_moves[best_move]
            if best_weight / total_weight > 0.4:  # If move has >40% of total weight
                return best_move

        # Otherwise weighted random selection
        if all_moves:
            moves = list(all_moves.keys())
            weights = [all_moves[move] / total_weight for move in moves]
            return random.choices(moves, weights=weights)[0]

        return None

    def evaluate_position(self, board: chess.Board) -> float:
        """Enhanced position evaluation"""
        if board.is_checkmate():
            return -np.inf if board.turn == chess.WHITE else np.inf
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Base material score
        material_score = sum(
            len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type] -
            len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        )

        # Development and center control
        development_score = self._evaluate_development(board)
        position_score = self._evaluate_position_strength(board)
        
        # Combine scores with weights
        final_score = (
            material_score + 
            development_score * 30 +  # Increased development importance
            position_score * 20
        ) / 100.0

        return final_score

    def _evaluate_development(self, board: chess.Board) -> float:
        """Evaluate piece development and center control"""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            
            # Piece development
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                developed = 0
                for square in board.pieces(piece_type, color):
                    rank = chess.square_rank(square)
                    # Check if piece has moved from back rank
                    if color == chess.WHITE and rank > 1:
                        developed += 1
                    elif color == chess.BLACK and rank < 6:
                        developed += 1
                score += developed * 0.5 * multiplier

            # Castling bonus
            if board.has_castling_rights(color):
                score += 0.8 * multiplier
                
            # Central pawn control
            for square in self.center_squares:
                if board.piece_at(square):
                    piece = board.piece_at(square)
                    if piece.piece_type == chess.PAWN and piece.color == color:
                        score += 0.4 * multiplier

            # Early queen penalties
            queen_moves = 0
            for move in board.move_stack[:8]:  # First 8 moves
                if board.piece_type_at(move.from_square) == chess.QUEEN:
                    queen_moves += 1
            score -= queen_moves * 0.3 * multiplier

        return score

    def _evaluate_position_strength(self, board: chess.Board) -> float:
        """Evaluate overall position strength"""
        score = 0.0

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            
            # Center control
            for square in self.center_squares:
                attackers = len(list(board.attackers(color, square)))
                score += attackers * 0.2 * multiplier
            
            # Extended center influence
            for square in self.extended_center:
                attackers = len(list(board.attackers(color, square)))
                score += attackers * 0.1 * multiplier

            # Pawn structure
            pawn_files = [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]
            isolated_pawns = sum(1 for f in pawn_files 
                               if f-1 not in pawn_files and f+1 not in pawn_files)
            doubled_pawns = sum(1 for f in set(pawn_files) 
                              if pawn_files.count(f) > 1)
            
            score -= (isolated_pawns * 0.2 + doubled_pawns * 0.3) * multiplier

        return score

    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, str]:
        """Get best move with explanation"""
        # Try book move first
        book_move = self.get_book_move(board)
        if book_move:
            return book_move, "Book move from opening theory"

        # Regular search for best move
        legal_moves = list(board.legal_moves)
        best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = legal_moves[0]

        for move in legal_moves:
            board.push(move)
            eval = self.evaluate_position(board)
            board.pop()

            if board.turn == chess.WHITE:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        if abs(best_eval) > 5:
            explanation = "Decisive advantage found"
        elif abs(best_eval) > 2:
            explanation = "Clear advantage gained"
        elif abs(best_eval) > 0.5:
            explanation = "Slight edge obtained"
        else:
            explanation = "Equal position maintained"

        return best_move, explanation