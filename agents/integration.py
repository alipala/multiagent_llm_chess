from typing import Dict, Optional
import logging
from .base import OrchestratorAgent, BoardAgent, ChessAgent
from .knowledge_base import ChessKnowledgeBase

logger = logging.getLogger(__name__)

class ChessAgentSystem:
    """Manages the chess agent system and integration with existing game logic"""
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.board_agent = BoardAgent()
        self.knowledge_base = ChessKnowledgeBase()
        
        # Register board agent with orchestrator
        self.orchestrator.register_agent("board", self.board_agent)
        
        self._initialize_system()
        logger.info("Chess agent system initialized")

    def _initialize_system(self):
        """Initialize the agent system with basic chess knowledge"""
        basic_knowledge = [
            """Chess openings principles:
            1. Control the center
            2. Develop pieces
            3. Castle early
            4. Connect rooks""",
            
            """Basic tactics:
            1. Pin
            2. Fork
            3. Double attack
            4. Discovered attack""",
            
            """Endgame principles:
            1. King activity
            2. Pawn structure
            3. Piece coordination
            4. Opposition"""
        ]
        
        metadata = [
            {"type": "opening", "category": "principles"},
            {"type": "tactics", "category": "basics"},
            {"type": "endgame", "category": "principles"}
        ]
        
        self.knowledge_base.add_knowledge(basic_knowledge, metadata)

    async def handle_move_request(self, board_fen: str) -> Dict:
        """Handle move request from the game system"""
        try:
            # Create game state message
            game_state = {
                "game_state": {
                    "fen": board_fen,
                    "turn": "w" if "w" in board_fen else "b"
                }
            }
            
            # Get response from orchestrator
            response = await self.orchestrator.process_message(game_state, None)
            
            return {
                "status": "success",
                "move": response.get("move"),
                "explanation": response.get("explanation"),
                "evaluation": response.get("evaluation", 0.0)
            }
        except Exception as e:
            logger.error(f"Error handling move request: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def make_move(self, move: str) -> Dict:
        """Execute a move in the game"""
        try:
            message = {"make_move": move}
            response = await self.board_agent.process_message(message, None)
            return response
        except Exception as e:
            logger.error(f"Error making move: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_game_state(self) -> Dict:
        """Get current game state"""
        return self.board_agent.get_game_state()

    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.knowledge_base.get_statistics()