const socket = io();
let board = null;
let game = new Chess();
let aiInterval = null;
let capturedPieces = { w: [], b: [] };

let gameStartTime = null;
let gameDurationInterval = null;

// Initialize evaluation state
let evalVisible = true;
let lastEvaluation = 0;

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

function onDrop(source, target) {
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateStatus();
    updateThinkingDots();
    socket.emit('make_move', { move: move.from + move.to });
}

function updateStatus() {
    let status = '';
    let moveColor = game.turn() === 'b' ? 'Black' : 'White';

    if (game.in_checkmate()) {
        status = `Game over, ${moveColor} is in checkmate.`;
    } else if (game.in_draw()) {
        status = 'Game over, drawn position';
    } else {
        status = `${moveColor} to move`;
        if (game.in_check()) {
            status += `, ${moveColor} is in check`;
        }
    }

    $('#status').text(status);
}

function updateEvaluation(evaluation) {
    if (!evalVisible) return;

    const fullScore = Math.max(Math.min(evaluation, 20), -20); // Clamp between -20 and 20
    const abbreviated = Math.abs(fullScore) >= 10 ? Math.round(fullScore) : fullScore.toFixed(1);
    lastEvaluation = fullScore;
    
    // Update score displays
    document.querySelector('.eval-score-full').textContent = 
        (fullScore > 0 ? '+' : '') + fullScore.toFixed(1);
    document.querySelector('.eval-score-abbreviated').textContent = 
        (fullScore > 0 ? '+' : '') + abbreviated;

    // Calculate position percentage (0-100)
    let percentage;
    if (Math.abs(fullScore) >= 5) {
        // After ±5, scale logarithmically
        percentage = 50 + (Math.sign(fullScore) * (45 + Math.min(Math.log2(Math.abs(fullScore) - 4), 5)));
    } else {
        // Linear scaling between -5 and +5
        percentage = 50 + (fullScore * 10);
    }

    // Update bar position
    const whiteBar = document.querySelector('.eval-color.white');
    whiteBar.style.transform = `translate3d(0, ${100 - percentage}%, 0)`;

    // Update colors based on advantage
    if (Math.abs(fullScore) < 0.2) {
        document.querySelector('.eval-color.draw').style.opacity = '1';
    } else {
        document.querySelector('.eval-color.draw').style.opacity = '0.3';
    }
}

function updateThinkingDots() {
    const currentTurn = game.turn();
    const whiteThinking = document.getElementById('white-thinking');
    const blackThinking = document.getElementById('black-thinking');
    
    // Hide both thinking indicators first
    whiteThinking.classList.remove('active');
    blackThinking.classList.remove('active');
    
    // Show thinking dots for current player
    if (!game.game_over()) {
        if (currentTurn === 'w') {
            whiteThinking.classList.add('active');
        } else {
            blackThinking.classList.add('active');
        }
    }
}

function updateMoveHistory(move, result) {
    const moveHistory = document.getElementById('move-history');
    const moveElement = document.createElement('div');
    moveElement.className = `move-item ${game.turn() === 'w' ? 'black-move' : 'white-move'}`;
    moveElement.textContent = `${game.turn() === 'w' ? 'Black' : 'White'}: ${move} (${result})`;
    moveHistory.appendChild(moveElement);
    moveHistory.scrollTop = moveHistory.scrollHeight;
}

function updateCapturedPieces(move) {
    const capturedPiece = game.history({ verbose: true }).slice(-1)[0].captured;
    if (capturedPiece) {
        const color = game.turn() === 'w' ? 'b' : 'w';
        capturedPieces[color].push(getPieceSymbol(capturedPiece));
        updateCapturedPiecesDisplay(color);
    }
}

function updateCapturedPiecesDisplay(color) {
    const container = $(`#${color === 'w' ? 'white' : 'black'}-captured`);
    container.empty();
    let currentRow = $('<div class="captured-row"></div>');
    container.append(currentRow);

    capturedPieces[color].forEach((piece, index) => {
        if (index > 0 && index % 20 === 0) {
            currentRow = $('<div class="captured-row"></div>');
            container.append(currentRow);
        }
        currentRow.append(`<span class="captured-piece">${piece}</span>`);
    });
}

function getPieceSymbol(piece) {
    const symbols = { p: '♙', n: '♘', b: '♗', r: '♖', q: '♕', k: '♔' };
    return symbols[piece.toLowerCase()] || '';
}

function startAIGame() {
    $('#start-ai-game').hide();
    $('#stop-ai-game').show();
    $('#ai-move').prop('disabled', true);
    gameStartTime = new Date();
    updateGameDuration();
    gameDurationInterval = setInterval(updateGameDuration, 1000);
    requestAIMove();
    aiInterval = setInterval(requestAIMove, 2000);
    
    // Hide game summary when starting new game
    document.querySelector('.game-summary-section').classList.remove('visible');
}

function stopAIGame() {
    $('#start-ai-game').show();
    $('#stop-ai-game').hide();
    $('#ai-move').prop('disabled', false);
    clearInterval(aiInterval);
    clearInterval(gameDurationInterval);
}

function requestAIMove() {
    if (!game.game_over()) {
        socket.emit('request_ai_move');
    } else {
        stopAIGame();
        showGameResult();
        requestGameSummary();
    }
}

function showGameResult() {
    let result = '';
    if (game.in_checkmate()) {
        result = `Checkmate! ${game.turn() === 'w' ? 'Black' : 'White'} wins.`;
    } else if (game.in_stalemate()) {
        result = 'Stalemate! The game is a draw.';
    } else if (game.in_threefold_repetition()) {
        result = 'Draw by threefold repetition.';
    } else if (game.insufficient_material()) {
        result = 'Draw due to insufficient material.';
    } else if (game.in_draw()) {
        result = 'The game is a draw.';
    }
    $('#game-result').text(result);
    
    // Show game summary section
    document.querySelector('.game-summary-section').classList.add('visible');
}

function resetGame() {
    game.reset();
    board.position('start');
    $('#move-history').empty();
    $('#move-explanation').empty();
    $('#game-summary').empty();
    capturedPieces = { w: [], b: [] };
    updateCapturedPiecesDisplay('w');
    updateCapturedPiecesDisplay('b');
    $('#game-result').text('');
    $('#game-duration').text('Game Duration: 00:00');
    clearInterval(gameDurationInterval);
    updateStatus();
    updateThinkingDots();
    stopAIGame();
    socket.emit('reset_game');
    
    // Reset evaluation
    updateEvaluation(0);
    
    // Hide game summary
    document.querySelector('.game-summary-section').classList.remove('visible');
}

function updateGameDuration() {
    if (gameStartTime) {
        const duration = new Date() - gameStartTime;
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        $('#game-duration').text(`Game Duration: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
    }
}

function requestGameSummary() {
    socket.emit('get_game_summary');
}

// Initialize board
const config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png'
};
board = Chessboard('board', config);

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('game_state', (data) => {
    game.load(data.fen);
    board.position(data.fen);
    updateStatus();
    updateThinkingDots();
});

socket.on('move_made', (data) => {
    const move = game.move({
        from: data.move.slice(0, 2),
        to: data.move.slice(2, 4),
        promotion: 'q'
    });

    if (move) {
        board.position(game.fen());
        updateStatus();
        updateThinkingDots();
        updateMoveHistory(data.move, data.result);
        updateCapturedPieces(move);
        $('#move-explanation').text(data.explanation);

        if (data.evaluation !== undefined) {
            updateEvaluation(data.evaluation);
        }
    }

    if (data.game_over) {
        stopAIGame();
        showGameResult();
        requestGameSummary();
    }
});

socket.on('game_summary', (data) => {
    $('#game-summary').text(data.summary);
    // Ensure the game summary section is visible
    document.querySelector('.game-summary-section').classList.add('visible');
});

socket.on('error', (data) => {
    console.error('Error:', data.message);
    alert('Error: ' + data.message);
});

socket.on('analysis_result', (data) => {
    if (data.analysis && data.analysis.evaluation !== undefined) {
        updateEvaluation(data.analysis.evaluation);
    }
});

// Event listeners
$(document).ready(function() {
    // Initialize buttons
    $('#start-ai-game').click(startAIGame);
    $('#stop-ai-game').click(stopAIGame);
    $('#ai-move').click(requestAIMove);
    $('#reset-game').click(resetGame);
    $('#export-pgn').click(() => {
        socket.emit('get_pgn');
    });
    
    // Evaluation toggle
    $('#toggle-eval').click(function() {
        evalVisible = !evalVisible;
        const evalContainer = $('.evaluation-container');
        if (evalVisible) {
            evalContainer.show();
            $(this).text('Hide Evaluation');
            updateEvaluation(lastEvaluation);
        } else {
            evalContainer.hide();
            $(this).text('Show Evaluation');
        }
    });

    // Add hover effect for abbreviated/full score
    $('.eval-bar-wrapper').hover(
        function() {
            $('.eval-score-full').hide();
            $('.eval-score-abbreviated').show();
        },
        function() {
            $('.eval-score-full').show();
            $('.eval-score-abbreviated').hide();
        }
    );

    // Initialize thinking dots
    updateThinkingDots();
});

// PGN Export handler
socket.on('pgn_data', (data) => {
    const blob = new Blob([data.pgn], {type: 'text/plain'});
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'chess_game.pgn';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
});

updateStatus();