import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
import chess
import numpy as np
import random
from .engine import ChessEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4Model:
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        logger.info(f"Initialized GPT4Model with temperature {temperature}")
        
    def explain_move(self, context: str) -> str:
        """Generate explanation for a chess move"""
        try:
            # Extract key information from context
            move_info = self._parse_context(context)
            
            # Generate tailored explanation based on move type and context
            if 'capture' in context.lower():
                piece_captured = self._get_captured_piece(context)
                return f"A tactical move capturing {piece_captured} and improving material balance."
            
            elif 'check' in context.lower():
                return f"A strong move putting the king in check and creating immediate threats."
            
            elif 'book move' in context.lower():
                return "A standard book move from chess theory, maintaining good piece coordination."
            
            elif 'evaluation:' in context.lower():
                eval_str = self._extract_evaluation(context)
                if eval_str:
                    eval_float = float(eval_str)
                    if abs(eval_float) > 2.0:
                        return f"A decisive move giving {'White' if eval_float > 0 else 'Black'} a significant advantage."
                    elif abs(eval_float) > 1.0:
                        return f"A strong move giving {'White' if eval_float > 0 else 'Black'} a clear advantage."
                    elif abs(eval_float) > 0.5:
                        return f"A good move giving {'White' if eval_float > 0 else 'Black'} a slight edge."
                    else:
                        return "An equal position with balanced chances for both sides."
                        
            elif 'development' in context.lower():
                return "A developing move improving piece coordination and control."
                
            # Analyze move characteristics
            explanation = self._analyze_move(move_info)
            if explanation:
                return explanation
                
            return "A positional move maintaining balance and creating opportunities."
            
        except Exception as e:
            logger.error(f"Error generating move explanation: {e}")
            return "Move selected based on position evaluation."
            
    def _parse_context(self, context: str) -> dict:
        """Extract relevant information from move context"""
        info = {}
        try:
            # Extract move
            if 'move' in context.lower():
                move_parts = context.split('move:')
                if len(move_parts) > 1:
                    info['move'] = move_parts[1].strip().split()[0]
                    
            # Extract piece type
            pieces = {'K': 'king', 'Q': 'queen', 'R': 'rook', 
                     'B': 'bishop', 'N': 'knight', 'P': 'pawn'}
            for symbol, piece in pieces.items():
                if symbol in context:
                    info['piece'] = piece
                    break
                    
            # Extract evaluation
            if 'evaluation:' in context.lower():
                eval_parts = context.split('evaluation:')
                if len(eval_parts) > 1:
                    info['evaluation'] = float(eval_parts[1].strip().split()[0])
                    
        except Exception as e:
            logger.error(f"Error parsing move context: {e}")
            
        return info
        
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
        
    def _analyze_move(self, move_info: dict) -> Optional[str]:
        """Analyze move characteristics and generate explanation"""
        try:
            if 'piece' in move_info and 'evaluation' in move_info:
                piece = move_info['piece']
                eval_val = move_info['evaluation']
                
                if abs(eval_val) > 1.5:
                    strength = "decisive"
                elif abs(eval_val) > 0.8:
                    strength = "strong"
                else:
                    strength = "solid"
                    
                return f"A {strength} {piece} move that {'gains advantage' if eval_val > 0 else 'maintains balance'}."
                
        except Exception as e:
            logger.error(f"Error analyzing move: {e}")
            
        return None

class ChessTransformer(nn.Module):
    def __init__(self, encoder_layers: int = 6, attention_heads: int = 8, d_model: int = 512):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, d_model)
        self.position_embedding = nn.Embedding(64, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=attention_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.move_predictor = nn.Linear(d_model, 4096)
        logger.info(f"Initialized ChessTransformer with {encoder_layers} layers")
        
    def analyze_position(self, fen: str) -> str:
        """Analyze chess position from FEN string"""
        try:
            board = chess.Board(fen)
            
            # Basic position analysis
            phase = "Opening" if board.fullmove_number < 10 else \
                   "Endgame" if len(board.pieces(chess.QUEEN, chess.WHITE)) + \
                              len(board.pieces(chess.QUEEN, chess.BLACK)) == 0 else \
                   "Middlegame"
            
            return f"Game phase: {phase}"
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            return "Position analysis unavailable"
    
    def evaluate_position(self, fen: str) -> float:
        """Evaluate position strength"""
        try:
            board = chess.Board(fen)
            
            # Simple material counting
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9
            }
            
            evaluation = 0
            for piece_type, value in piece_values.items():
                evaluation += len(board.pieces(piece_type, chess.WHITE)) * value
                evaluation -= len(board.pieces(piece_type, chess.BLACK)) * value
            
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating position: {e}")
            return 0.0

class HybridArchitecture:
    def __init__(self, llm, chess_transformer):
        self.llm = llm
        self.chess_transformer = chess_transformer
        self.chess_engine = ChessEngine()
        logger.info("Hybrid Architecture initialized with all components")

    def get_move(self, board: chess.Board, legal_moves: List[str]) -> Tuple[str, str]:
        """Get best move and explanation"""
        try:
            # Try book move first
            book_move = self.chess_engine.get_book_move(board)
            if book_move:
                context = (f"Book move: {book_move.uci()}\n"
                        f"Position FEN: {board.fen()}\n"
                        f"This is a common theoretical position.")
                explanation = self.llm.explain_move(context)
                return book_move.uci(), explanation

            # Get engine move
            engine_move, eval_str = self.chess_engine.get_best_move(board)
            
            # Build rich context for move explanation
            context = []
            context.append(f"Move: {engine_move.uci()}")
            context.append(f"Piece: {board.piece_at(engine_move.from_square)}")
            context.append(f"Position FEN: {board.fen()}")
            context.append(f"Evaluation: {eval_str}")
            
            # Add special move information
            if board.is_capture(engine_move):
                captured = board.piece_at(engine_move.to_square)
                context.append(f"Capture: takes {captured}")
            if board.is_check():
                context.append("Move gives check")
            if board.is_castling(engine_move):
                context.append("Castling move")
                
            # Add phase information
            if board.fullmove_number <= 10:
                context.append("Opening phase")
            elif len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) == 0:
                context.append("Endgame phase")
            else:
                context.append("Middlegame phase")
                
            explanation = self.llm.explain_move("\n".join(context))
            return engine_move.uci(), explanation

        except Exception as e:
            logger.error(f"Error in get_move: {e}")
            fallback_move = chess.Move.from_uci(legal_moves[0])
            return legal_moves[0], "Fallback move selected due to error"

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate chess position"""
        try:
            # Combine engine and transformer evaluations
            engine_eval = self.chess_engine.evaluate_position(board)
            transformer_eval = self.chess_transformer.evaluate_position(board.fen())
            
            # Weight engine evaluation more heavily
            return engine_eval * 0.8 + transformer_eval * 0.2
        except Exception as e:
            logger.error(f"Error in evaluate_position: {e}")
            # Fallback to engine evaluation only
            return self.chess_engine.evaluate_position(board)

    def _get_material_count(self, board: chess.Board) -> float:
        """
        Fallback method to get basic material count
        Args:
            board: Current chess board position
        Returns:
            Float representing material balance
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Not counted in material balance
        }
        
        material_balance = 0
        for piece_type in piece_values:
            material_balance += (
                len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type] -
                len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            )
        return material_balance

    def cleanup(self):
        """
        Cleanup resources when shutting down
        """
        try:
            if hasattr(self.chess_engine, 'cleanup'):
                self.chess_engine.cleanup()
            logger.info("Hybrid Architecture cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")