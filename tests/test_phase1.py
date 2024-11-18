import asyncio
import logging
from agents.integration import ChessAgentSystem
import chess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
print(f"API Key set: {'OPENAI_API_KEY' in os.environ}")

async def test_agent_system():
    """Test the chess agent system implementation"""
    
    # Initialize the system
    system = ChessAgentSystem()
    logger.info("Agent system initialized")
    
    # Test knowledge base
    stats = system.get_knowledge_stats()
    logger.info(f"Knowledge base statistics: {stats}")
    
    # Test game state
    state = system.get_game_state()
    logger.info(f"Initial game state: {state}")
    
    # Test making moves
    test_moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    for move in test_moves:
        result = await system.make_move(move)
        logger.info(f"Move {move} result: {result}")
        
        # Get updated state
        state = system.get_game_state()
        logger.info(f"Game state after {move}: {state}")
    
    # Test move request
    response = await system.handle_move_request(state["fen"])
    logger.info(f"Move request response: {response}")

if __name__ == "__main__":
    asyncio.run(test_agent_system())