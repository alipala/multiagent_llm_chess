import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
import chess
import numpy as np
import random
from langchain_openai import ChatOpenAI 


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessEngine:
    def __init__(self, temperature: float = 0.7):
        self.chat = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature
        )
        self.logger = logging.getLogger(__name__)

    def _create_move_prompt(self, board: chess.Board, legal_moves: List[str]) -> str:
        """Create a detailed prompt for move selection"""
        position = self._analyze_position(board)
        turn = "White" if board.turn else "Black"
        
        return f"""
        As a {turn} chess player, analyze this position and select the best move:
        
        Position Analysis:
        {position}
        
        Legal moves: {', '.join(legal_moves)}
        
        Consider:
        1. Piece development and coordination
        2. Center control and space advantage
        3. King safety and pawn structure
        4. Tactical opportunities
        
        In the opening phase:
        - Control center with e4/d4 (or e5/d5 for Black)
        - Develop knights to good squares (f3/c3 or f6/c6)
        - Don't move same piece twice
        - Don't bring queen out early
        - Castle within first 7-8 moves

        Return your response in format:
        Move: <uci_move>
        Explanation: <your_explanation>
        """
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Calculate material balance"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.2,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0
        for piece_type in piece_values:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += piece_values[piece_type] * (white_count - black_count)
            
        return score

    def _evaluate_positional_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors"""
        score = 0
        
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                score += 0.3 if piece.color == chess.WHITE else -0.3
                
        # King safety
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        white_attackers = len(list(board.attackers(chess.BLACK, white_king_sq))) if white_king_sq else 0
        black_attackers = len(list(board.attackers(chess.WHITE, black_king_sq))) if black_king_sq else 0
        score -= 0.2 * (white_attackers - black_attackers)
        
        return score

    def _analyze_position(self, board: chess.Board) -> str:
        """Create a string representation of key position features"""
        analysis = []
        
        # Material count
        piece_map = {
            'P': len(board.pieces(chess.PAWN, chess.WHITE)),
            'N': len(board.pieces(chess.KNIGHT, chess.WHITE)),
            'B': len(board.pieces(chess.BISHOP, chess.WHITE)),
            'R': len(board.pieces(chess.ROOK, chess.WHITE)),
            'Q': len(board.pieces(chess.QUEEN, chess.WHITE)),
            'p': len(board.pieces(chess.PAWN, chess.BLACK)),
            'n': len(board.pieces(chess.KNIGHT, chess.BLACK)),
            'b': len(board.pieces(chess.BISHOP, chess.BLACK)),
            'r': len(board.pieces(chess.ROOK, chess.BLACK)),
            'q': len(board.pieces(chess.QUEEN, chess.BLACK))
        }
        
        analysis.append(f"Material: {piece_map}")
        
        # King safety
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        analysis.append(f"White king: {chess.square_name(white_king_sq)}")
        analysis.append(f"Black king: {chess.square_name(black_king_sq)}")
        
        # Game phase
        move_num = board.fullmove_number
        phase = "opening" if move_num <= 10 else "middlegame" if move_num <= 30 else "endgame"
        analysis.append(f"Game phase: {phase}")
        
        return "; ".join(analysis)

    def evaluate_position(self, board: chess.Board) -> float:
        """Complete position evaluation"""
        if board.is_checkmate():
            return 999 if board.turn == chess.WHITE else -999
            
        material_score = self._evaluate_material(board)
        positional_score = self._evaluate_positional_factors(board)
        
        # Combine scores with weights
        total_score = material_score + 0.5 * positional_score
            
        return total_score
    
    def _validate_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Validate move against basic principles"""
        
        # Opening phase checks (moves 1-10)
        if board.fullmove_number <= 10:
            piece = board.piece_at(move.from_square)
            
            # Don't move same piece twice in opening
            if board.fullmove_number <= 5:
                for past_move in board.move_stack[-2:]:
                    if past_move.from_square == move.from_square:
                        return False
                        
            # Don't bring queen out too early
            if piece.piece_type == chess.QUEEN and board.fullmove_number < 7:
                return False
                
            # Don't move edge pawns early
            if piece.piece_type == chess.PAWN:
                if chess.square_file(move.from_square) in [0, 7] and board.fullmove_number < 8:
                    return False
        
        # General safety checks
        future_board = board.copy()
        future_board.push(move)
        
        # Don't allow moves that hang pieces
        if len(future_board.attackers(not board.turn, move.to_square)) > len(future_board.attackers(board.turn, move.to_square)):
            return False
            
        # Don't weaken king safety
        king_square = future_board.king(board.turn)
        if len(future_board.attackers(not board.turn, king_square)) > 0:
            return False
            
        return True

    def get_move(self, board: chess.Board, legal_moves: List[str]) -> Tuple[str, str]:
        try:
            # Create position analysis
            position = self._analyze_position(board)
            
            # Get response from GPT-4
            response = self.chat.predict(self._create_move_prompt(board, legal_moves))
            
            # Parse response
            move = legal_moves[0]  # default
            explanation = "Default move selected"
            
            for line in response.split('\n'):
                if line.startswith('Move:'):
                    candidate = line.split(':')[1].strip()
                    if candidate in legal_moves:
                        # Validate move
                        chess_move = chess.Move.from_uci(candidate)
                        if self._validate_move(board, chess_move):
                            move = candidate
                            
                elif line.startswith('Explanation:'):
                    explanation = line.split(':')[1].strip()
            
            return move, explanation
            
        except Exception as e:
            self.logger.error(f"Error in move generation: {str(e)}")
            return legal_moves[0], "Fallback move selected"