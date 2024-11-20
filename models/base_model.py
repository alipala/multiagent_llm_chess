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

class GPT4Model:
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.chat = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature
        )
        logger.info(f"Initialized GPT4Model with temperature {temperature}")

    def generate_commentary(self, context: str) -> str:
        try:
            # Use predict method for generating commentary
            response = self.chat.predict(context)
            return response
        except Exception as e:
            logger.error(f"Error in GPT-4 commentary generation: {str(e)}")
            return f"{context.split()[-1].capitalize()} move focusing on position control."

    def encode_position(self, board: chess.Board) -> str:
        return f"""
        Position FEN: {board.fen()}
        Material count: {self._get_material_count(board)}
        King safety: {self._analyze_king_safety(board)}
        Center control: {self._analyze_center_control(board)}
        """
    
    def _get_material_count(self, board: chess.Board) -> Dict[str, int]:
        material = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material[piece.symbol().upper()] += 1
        return material
    
    def _analyze_king_safety(self, board: chess.Board) -> str:
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        return f"White king on {chess.square_name(white_king_square)}, Black king on {chess.square_name(black_king_square)}"
    
    def _analyze_center_control(self, board: chess.Board) -> str:
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = {'white': 0, 'black': 0}
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                control['white'] += 1
            if board.is_attacked_by(chess.BLACK, square):
                control['black'] += 1
        return f"White controls {control['white']} center squares, Black controls {control['black']}"

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
        
    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        if board_state.dim() == 1:
            board_state = board_state.unsqueeze(0)
        piece_emb = self.piece_embedding(board_state)
        pos_ids = torch.arange(64, device=board_state.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = piece_emb + pos_emb
        x = self.transformer(x)
        return self.move_predictor(x.mean(dim=1))

class HybridArchitecture:
    def __init__(self, llm: GPT4Model, chess_transformer: ChessTransformer, integration_layer: Optional[nn.Module] = None):
        self.llm = llm
        self.chess_transformer = chess_transformer
        self.integration_layer = integration_layer
        self.piece_to_index = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
            '.': 0, None: 0
        }
        logger.info("Initialized HybridArchitecture")

    def get_move(self, board: chess.Board, legal_moves: List[str]) -> Tuple[str, str]:
        try:
            if not legal_moves:
                raise ValueError("No legal moves available")

            # Score all legal moves
            move_scores = {}
            for move in legal_moves:
                # Multiple evaluations to reduce randomness impact
                scores = [self._evaluate_move(board, move) for _ in range(3)]
                move_scores[move] = sum(scores) / 3

            # Select from top 3 moves randomly
            top_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            selected_move = random.choice(top_moves)[0]
            
            explanation = self._generate_explanation(board, selected_move)
            logger.info(f"Generated move {selected_move} with explanation: {explanation}")
            
            return selected_move, explanation
                
        except Exception as e:
            logger.error(f"Error in move generation: {str(e)}")
            return random.choice(legal_moves), "Move selected randomly due to error"

    def _evaluate_move(self, board: chess.Board, move: str) -> float:
        try:
            chess_move = chess.Move.from_uci(move)
            score = 0.0
            
            # Add randomization factor (0-0.2)
            score += random.uniform(0, 0.2)
            
            # Piece value factor
            piece = board.piece_at(chess_move.from_square)
            if piece:
                piece_values = {'P': 1, 'N': 3, 'B': 3.2, 'R': 5, 'Q': 9, 'K': 0}
                score += piece_values.get(piece.symbol().upper(), 0) * 0.1

            # Center control with randomized bonus
            to_square = chess_move.to_square
            central_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
            if to_square in central_squares:
                score += 0.3 + random.uniform(0, 0.4)
            
            # Extended center bonus
            extended_center = {chess.C3, chess.D3, chess.E3, chess.F3, 
                            chess.C6, chess.D6, chess.E6, chess.F6}
            if to_square in extended_center:
                score += 0.2 + random.uniform(0, 0.2)

            # Capture value with randomization
            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                if captured_piece:
                    score += piece_values.get(captured_piece.symbol().upper(), 0)
                    score += random.uniform(0, 1)  # Random bonus for captures

            # Development bonus for early game
            if board.fullmove_number < 10:
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 0.4 + random.uniform(0, 0.3)

            # Pawn structure consideration
            if piece and piece.piece_type == chess.PAWN:
                # Bonus for advancing center pawns
                if chess_move.from_square in [chess.E2, chess.D2, chess.E7, chess.D7]:
                    score += 0.3 + random.uniform(0, 0.2)

            return score
                
        except Exception as e:
            logger.error(f"Error in move evaluation: {str(e)}")
            return random.uniform(0, 1)  # Return random score on error

    def _generate_explanation(self, board: chess.Board, move: str) -> str:
        """Generate chess commentary using HybridArchitecture context"""
        try:
            chess_move = chess.Move.from_uci(move)
            
            # Get board state and pieces
            piece = board.piece_at(chess_move.from_square)
            piece_name = chess.piece_name(piece.piece_type) if piece else "piece"
            from_square = chess.square_name(chess_move.from_square)
            to_square = chess.square_name(chess_move.to_square)
            
            # Create a copy of the board and make the move for evaluation
            board_copy = board.copy()
            board_copy.push(chess_move)
            position_eval = self.evaluate_position(board_copy)
            
            # Additional context for the commentary
            context = {
                "move_number": board.fullmove_number,
                "is_capture": board.is_capture(chess_move),
                "captured_piece": chess.piece_name(board_copy.piece_at(chess_move.to_square).piece_type) if board.is_capture(chess_move) else None,
                "gives_check": board.gives_check(chess_move),
                "controls_center": to_square in ['e4', 'd4', 'e5', 'd5'],
                "evaluation": position_eval,
                "phase": "Opening" if board.fullmove_number <= 10 else "Middlegame" if board.fullmove_number <= 30 else "Endgame"
            }
            
            prompt = f"""As a chess commentator, analyze this move in the {context['phase']}:
            {piece_name.title()} moves from {from_square} to {to_square}
            {"Captures " + context['captured_piece'] + "!" if context['is_capture'] else ""}
            {"Delivers check!" if context['gives_check'] else ""}
            {"Controls center!" if context['controls_center'] else ""}
            Position evaluation: {context['evaluation']:.2f}
            
            Provide brief, exciting commentary focusing on tactical and strategic implications."""
            
            commentary = self.llm.generate_commentary(prompt)
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Move selected based on position evaluation."
        

    def evaluate_position(self, board: chess.Board) -> float:

        try:
            # Material evaluation
            material_score = self._evaluate_material(board)
            
            # Position evaluation
            position_score = self._evaluate_positional_factors(board)
            
            # Combine scores with proper weighting
            total_score = (
                0.6 * material_score +  # Material is most important
                0.4 * position_score    # Position factors
            )
            
            # Normalize score to reasonable range (-10 to 10)
            normalized_score = max(min(total_score / 100, 10), -10)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error in position evaluation: {str(e)}")
            return 0.0

    def _evaluate_material(self, board: chess.Board) -> float:
        """Calculate material balance"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
                    
        return score

    def _evaluate_positional_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors"""
        score = 0
        
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                score += 10 if piece.color == chess.WHITE else -10
                
        # Piece mobility
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                mobility = len(list(board.attacks(square)))
                value = mobility * 2
                score += value if piece.color == chess.WHITE else -value
                
        # King safety
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                attackers = len(list(board.attackers(not color, king_square)))
                safety_penalty = attackers * 10
                score += -safety_penalty if color == chess.WHITE else safety_penalty
                
        return score