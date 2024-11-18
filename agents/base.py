from typing import Dict, List, Optional, Any
import logging
import autogen
from chromadb import PersistentClient, Collection
from chromadb.config import Settings
import chess
import os
import json

logger = logging.getLogger(__name__)

class ChessAgent(autogen.AssistantAgent):
    """Base class for all chess agents"""

    def __init__(self, name: str, system_message: str, **kwargs):
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4",  # Explicitly specify the model
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            ],
            "temperature": 0.7,
            "request_timeout": 120,
        }
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
        self.termination_msg = "TASK_COMPLETE"
        logger.info(f"Initialized {name} agent")

    async def process_message(self, message: str, sender: Any) -> Dict:
        """Process incoming message and return response"""
        try:
            response = await self._process_message(message, sender)
            return response if response else {"status": "complete", "message": self.termination_msg}
        except Exception as e:
            logger.error(f"Error processing message in {self.name}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _process_message(self, message: str, sender: Any) -> Optional[Dict]:
        """Override this method in subclasses to implement specific agent behavior"""
        raise NotImplementedError

class OrchestratorAgent(ChessAgent):
    """Coordinates interactions between agents and manages game flow"""
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            system_message="""You are the orchestrator agent responsible for coordinating the chess game.
            Your tasks include:
            1. Managing communication between agents
            2. Tracking game state
            3. Delegating tasks to appropriate agents
            4. Ensuring proper game flow
            """
        )
        self.board_agent = None
        self.current_game_state = None
        self.active_agents = {}

    def register_agent(self, agent_type: str, agent: ChessAgent):
        """Register an agent with the orchestrator"""
        self.active_agents[agent_type] = agent
        if agent_type == "board":
            self.board_agent = agent
        logger.info(f"Registered {agent_type} agent")

    async def _process_message(self, message: Dict, sender: Any) -> Dict:
        """Process messages and coordinate responses"""
        try:
            if isinstance(message, dict):
                if "game_state" in message:
                    self.current_game_state = message["game_state"]
                    if self.board_agent:
                        board_response = await self.board_agent.process_message(
                            {"get_move": self.current_game_state}, 
                            self
                        )
                        
                        return {
                            "status": "success",
                            "move": board_response.get("move", ""),
                            "explanation": board_response.get("explanation", "Move selected based on position"),
                            "evaluation": board_response.get("evaluation", 0.0)
                        }
            
            return {
                "status": "error",
                "move": "",
                "explanation": "Invalid request format",
                "evaluation": 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            return {
                "status": "error",
                "move": "",
                "explanation": f"Error: {str(e)}",
                "evaluation": 0.0
            }

class BoardAgent(ChessAgent):
    """Manages game state and enforces rules"""
    
    def __init__(self):
        super().__init__(
            name="Board",
            system_message="""You are the board agent responsible for:
            1. Maintaining game state
            2. Validating moves
            3. Enforcing chess rules
            4. Tracking game statistics
            """
        )
        self.board = chess.Board()
        self.move_history = []
        self.statistics = {
            "captures": 0,
            "checks": 0,
            "castlings": 0
        }

    async def _process_message(self, message: Dict, sender: Any) -> Dict:
        """Process board-related messages"""
        try:
            if isinstance(message, dict):
                if "get_move" in message:
                    # Update internal board state from FEN if provided
                    if isinstance(message["get_move"], dict) and "fen" in message["get_move"]:
                        self.board.set_fen(message["get_move"]["fen"])
                    
                    legal_moves = self._get_legal_moves()
                    if legal_moves:
                        selected_move = self._select_move(legal_moves)
                        return {
                            "status": "success",
                            "move": selected_move,
                            "explanation": "Move selected based on position evaluation",
                            "evaluation": self._evaluate_position()
                        }
                    return {
                        "status": "error",
                        "message": "No legal moves available"
                    }
                    
                elif "make_move" in message:
                    return self._handle_move(message["make_move"])
            
            return {
                "status": "error",
                "message": "Invalid message format"
            }
            
        except Exception as e:
            logger.error(f"Error in board agent: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _handle_move(self, move: str) -> Dict:
        """Handle move execution and validation"""
        try:
            # Handle reset command
            if move == 'reset':
                self.board.reset()
                self.move_history = []
                self.statistics = {"captures": 0, "checks": 0, "castlings": 0}
                return {
                    "status": "success",
                    "fen": self.board.fen(),
                    "is_game_over": False,
                    "statistics": self.statistics
                }

            chess_move = chess.Move.from_uci(move)
            if chess_move in self.board.legal_moves:
                # Track statistics before making move
                is_capture = self.board.is_capture(chess_move)
                is_check = self.board.gives_check(chess_move)
                is_castling = self.board.is_castling(chess_move)
                
                # Make the move
                self.board.push(chess_move)
                self.move_history.append(move)
                
                # Update statistics
                if is_capture:
                    self.statistics["captures"] += 1
                if is_check:
                    self.statistics["checks"] += 1
                if is_castling:
                    self.statistics["castlings"] += 1
                
                return {
                    "status": "success",
                    "fen": self.board.fen(),
                    "is_game_over": self.board.is_game_over(),
                    "statistics": self.statistics
                }
            return {
                "status": "error",
                "message": f"Illegal move: {move}"
            }
        except ValueError as e:
            return {"status": "error", "message": f"Invalid move format: {move}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _select_move(self, legal_moves: List[str]) -> str:
        """Select a move based on basic strategy"""
        try:
            # Convert UCI moves to chess.Move objects
            moves = [chess.Move.from_uci(m) for m in legal_moves]
            
            # Basic strategy: prioritize center control and development
            center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
            development_pieces = {chess.KNIGHT, chess.BISHOP}
            
            # Score each move
            move_scores = []
            for move in moves:
                score = 0
                # Bonus for controlling center
                if move.to_square in center_squares:
                    score += 1
                # Bonus for developing pieces
                piece = self.board.piece_at(move.from_square)
                if piece and piece.piece_type in development_pieces:
                    score += 1
                move_scores.append((move, score))
            
            # Select the highest scoring move
            best_move = max(move_scores, key=lambda x: x[1])[0]
            return best_move.uci()
            
        except Exception as e:
            logger.error(f"Error selecting move: {str(e)}")
            return legal_moves[0]  # Fallback to first legal move

    def _get_legal_moves(self) -> List[str]:
        """Get list of legal moves in current position"""
        return [move.uci() for move in self.board.legal_moves]

    def get_game_state(self) -> Dict:
        """Get current game state"""
        return {
            "fen": self.board.fen(),
            "is_game_over": self.board.is_game_over(),
            "legal_moves": self._get_legal_moves(),
            "statistics": self.statistics,
            "move_history": self.move_history
        }
    
    def _evaluate_position(self) -> float:
        """Evaluate the current position"""
        try:
            if self.board.is_checkmate():
                return float('-inf') if self.board.turn else float('inf')
            
            # Material counting
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9
            }
            
            material_score = 0
            for piece_type in piece_values:
                material_score += len(self.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
                material_score -= len(self.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            
            # Position evaluation
            position_score = 0
            center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
            
            # Bonus for controlling center
            for square in center_squares:
                piece = self.board.piece_at(square)
                if piece:
                    bonus = 0.5 if piece.color == chess.WHITE else -0.5
                    position_score += bonus
            
            # Combine scores
            final_score = material_score + position_score * 0.1
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in position evaluation: {str(e)}")
            return 0.0