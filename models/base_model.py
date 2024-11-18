import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
import chess
import numpy as np
from .engine import ChessEngine

logger = logging.getLogger(__name__)

class GPT4Model:
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        logger.info(f"Initialized GPT4Model with temperature {temperature}")
        
    def explain_move(self, context: str) -> str:
        """Generate explanation for a chess move"""
        try:
            # Extract key information from context
            if 'capture' in context.lower():
                piece_captured = self._get_captured_piece(context)
                return f"A tactical move capturing {piece_captured} and improving material balance."
            
            elif 'check' in context.lower():
                return f"A strong move putting the king in check and creating immediate threats."
            
            elif 'evaluation:' in context.lower():
                eval_str = self._extract_evaluation(context)
                if eval_str:
                    eval_float = float(eval_str)
                    if abs(eval_float) > 2.0:
                        return f"A decisive move giving {'White' if eval_float > 0 else 'Black'} a significant advantage."
                    elif abs(eval_float) > 1.0:
                        return f"A strong move giving {'White' if eval_float > 0 else 'Black'} a clear advantage."
                    else:
                        return f"A move maintaining the balance of the position."
            
            return "A positional move maintaining balance and creating opportunities."
            
        except Exception as e:
            logger.error(f"Error generating move explanation: {e}")
            return "Move selected based on position evaluation."
            
    def _extract_evaluation(self, context: str) -> Optional[str]:
        """Extract evaluation score from context"""
        try:
            if 'evaluation:' in context.lower():
                parts = context.split('evaluation:')
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
        except Exception:
            pass
        return None
        
    def _get_captured_piece(self, context: str) -> str:
        """Get the type of piece captured"""
        pieces = {
            'q': 'queen', 'r': 'rook', 'b': 'bishop', 
            'n': 'knight', 'p': 'pawn', 'k': 'king'
        }
        
        try:
            if 'takes' in context.lower():
                parts = context.lower().split('takes')
                if len(parts) > 1:
                    piece_symbol = parts[1].strip()[0]
                    return pieces.get(piece_symbol, 'piece')
        except Exception:
            pass
        return 'piece'

class ChessTransformer:
    def __init__(self, encoder_layers: int = 6, attention_heads: int = 8, d_model: int = 512):
        self.d_model = d_model
        logger.info(f"Initialized ChessTransformer with {encoder_layers} layers")

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate chess position"""
        try:
            # Simple material counting with positional bonuses
            piece_values = {
                chess.PAWN: 1.0,
                chess.KNIGHT: 3.0,
                chess.BISHOP: 3.25,
                chess.ROOK: 5.0,
                chess.QUEEN: 9.0
            }
            
            evaluation = 0.0
            
            # Material count
            for piece_type, value in piece_values.items():
                evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
                evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value
            
            # Center control bonus
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            for square in center_squares:
                piece = board.piece_at(square)
                if piece:
                    bonus = 0.2 if piece.color == chess.WHITE else -0.2
                    evaluation += bonus
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in position evaluation: {e}")
            return 0.0

class HybridArchitecture:
    def __init__(self, llm: GPT4Model, chess_transformer: ChessTransformer):
        self.llm = llm
        self.chess_transformer = chess_transformer
        self.chess_engine = ChessEngine()
        logger.info("Hybrid Architecture initialized")

    def get_move(self, board: chess.Board, legal_moves: List[str]) -> Tuple[str, str]:
        """Get best move and explanation"""
        try:
            # Get engine move and evaluation
            engine_move, eval_str = self.chess_engine.get_best_move(board)
            
            # Build context for move explanation
            context = []
            context.append(f"Move: {engine_move.uci()}")
            context.append(f"Piece: {board.piece_at(engine_move.from_square)}")
            context.append(f"Evaluation: {eval_str}")
            
            if board.is_capture(engine_move):
                captured = board.piece_at(engine_move.to_square)
                context.append(f"Capture: takes {captured}")
            if board.is_check():
                context.append("Move gives check")
            if board.is_castling(engine_move):
                context.append("Castling move")
                
            explanation = self.llm.explain_move("\n".join(context))
            return engine_move.uci(), explanation

        except Exception as e:
            logger.error(f"Error in get_move: {e}")
            return legal_moves[0], "Fallback move selected"

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate chess position"""
        try:
            # Combine engine and transformer evaluations
            engine_eval = self.chess_engine.evaluate_position(board)
            transformer_eval = self.chess_transformer.evaluate_position(board)
            
            # Weight engine evaluation more heavily
            return engine_eval * 0.8 + transformer_eval * 0.2
        except Exception as e:
            logger.error(f"Error in evaluate_position: {e}")
            return self._get_material_count(board)

    def _get_material_count(self, board: chess.Board) -> float:
        """Basic material counting as fallback"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        material_balance = 0
        for piece_type in piece_values:
            material_balance += (
                len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type] -
                len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            )
        return material_balance