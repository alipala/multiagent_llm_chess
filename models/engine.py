import chess
import random
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ChessEngine:
    def __init__(self):
        self.OPENING_MOVES = 15  # First 15 moves considered opening phase
        
        # Simplified piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Important squares for position evaluation
        self.center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        self.extended_center = [chess.E3, chess.E6, chess.D3, chess.D6, 
                              chess.C3, chess.C4, chess.C5, chess.C6,
                              chess.F3, chess.F4, chess.F5, chess.F6]
        logger.info("ChessEngine initialized")

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate chess position"""
        if board.is_checkmate():
            return float('-inf') if board.turn == chess.WHITE else float('inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Material score
        material_score = sum(
            len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type] -
            len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        )

        # Position score
        position_score = self._evaluate_position_strength(board)
        
        # Combine scores
        final_score = (material_score + position_score * 20) / 100.0
        return final_score

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
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

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

        # Generate explanation based on evaluation
        if abs(best_eval) > 5:
            explanation = "Decisive advantage found"
        elif abs(best_eval) > 2:
            explanation = "Clear advantage gained"
        elif abs(best_eval) > 0.5:
            explanation = "Slight edge obtained"
        else:
            explanation = "Equal position maintained"

        return best_move, explanation