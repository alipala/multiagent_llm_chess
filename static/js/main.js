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
    const whiteThinking = document.getElementById('white-thinking');
    const blackThinking = document.getElementById('black-thinking');
    
    // Hide both thinking indicators by default
    whiteThinking.classList.remove('active');
    blackThinking.classList.remove('active');
    
    // Only show thinking dots if the game is in progress (aiInterval is active)
    if (aiInterval && !game.game_over()) {
        if (game.turn() === 'w') {
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
    
    // Update thinking dots after starting the game
    updateThinkingDots();
    
    // Hide game summary when starting new game
    document.querySelector('.game-summary-section').classList.remove('visible');
}

function stopAIGame() {
    $('#start-ai-game').show();
    $('#stop-ai-game').hide();
    $('#ai-move').prop('disabled', false);
    clearInterval(aiInterval);
    aiInterval = null; // Clear the interval reference
    clearInterval(gameDurationInterval);
    
    // Hide thinking dots when stopping the game
    updateThinkingDots();
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
    
    // Prepare summary section
    const summarySection = document.querySelector('.game-summary-section');
    if (summarySection) {
        summarySection.style.display = 'block';
        summarySection.classList.remove('visible');
    }
    
    // Request summary after a short delay to ensure proper display
    setTimeout(() => {
        requestGameSummary();
    }, 100);
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
    
    // Stop AI game if it's running
    if (aiInterval) {
        stopAIGame();
    }
    
    // Update thinking dots after reset
    updateThinkingDots();
    
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

function updateMoveExplanation(explanation) {
    const moveExplanation = document.getElementById('move-explanation');
    moveExplanation.textContent = ''; // Clear previous content
    moveExplanation.className = '';
    moveExplanation.classList.remove('new-comment');
    
    // Force a reflow to restart animation
    void moveExplanation.offsetWidth;
    
    moveExplanation.textContent = explanation;
    moveExplanation.style.animation = 'none';
    void moveExplanation.offsetWidth; // Trigger reflow
    moveExplanation.style.animation = 'fadeInUp 0.5s ease-out';
}

// Initialize board
const config = {    
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
    boardTheme: {
        light: '#e8eaed',     // Light squares
        dark: '#7fa650'       // Dark squares
    },
    responsive: true,
};
board = Chessboard('board', config);

window.addEventListener('resize', () => {
    board.resize();
});

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

    if (data.explanation) {
        updateMoveExplanation(data.explanation);
    }
});

socket.on('game_summary', (data) => {
    console.log('Received game summary:', data); // Debug log
    
    const summarySection = document.querySelector('.game-summary-section');
    const summaryContent = document.getElementById('game-summary');
    
    if (!summarySection || !summaryContent) {
        console.error('Summary elements not found');
        return;
    }
    
    // Clear previous content
    summaryContent.textContent = '';
    
    if (data.error) {
        console.error('Summary error:', data.error);
        summaryContent.textContent = 'Unable to generate game summary.';
        return;
    }

    // Create and append new summary with animation
    const summaryText = document.createElement('div');
    summaryText.className = 'summary-content';
    summaryText.textContent = data.summary;
    
    // Add statistics if available
    if (data.statistics) {
        const stats = document.createElement('div');
        stats.className = 'summary-statistics';
        stats.innerHTML = `
            <br>
            <strong>Game Statistics:</strong><br>
            Total Moves: ${data.statistics.total_moves}<br>
            Captures: ${data.statistics.captures}<br>
            Checks: ${data.statistics.checks}<br>
            Average Move Time: ${data.statistics.average_time.toFixed(2)}s
        `;
        summaryText.appendChild(stats);
    }

    // Apply fade-in animation
    summaryText.style.animation = 'fadeInUp 0.5s ease-out';
    
    // Append to container
    summaryContent.appendChild(summaryText);
    
    // Make sure the section is visible first
    summarySection.style.display = 'block';
    
    // Force a reflow
    void summarySection.offsetHeight;
    
    // Add visible class for animation
    summarySection.classList.add('visible');
    
    // Log visibility state
    console.log('Summary section display:', summarySection.style.display);
    console.log('Summary section visibility class:', summarySection.classList.contains('visible'));
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
});

updateStatus();

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

const additionalStyles = `
    .game-summary-section {
        display: none; /* Start hidden */
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.3s ease-out, transform 0.3s ease-out;
        margin-top: 20px;
        padding: 15px;
        background: rgba(35, 35, 35, 0.95);
        border-radius: 8px;
    }

    .game-summary-section.visible {
        opacity: 1;
        transform: translateY(0);
        display: block !important;
    }

    #game-summary {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
    }
`;

const style = document.createElement('style');
style.textContent = `
    .game-summary-section {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.3s ease-out, transform 0.3s ease-out;
        background: rgba(35, 35, 35, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }

    .game-summary-section.visible {
        opacity: 1;
        transform: translateY(0);
    }

    .summary-content {
        line-height: 1.6;
        color: #fff;
    }

    .summary-statistics {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9em;
        color: #ddd;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
style.textContent = additionalStyles + style.textContent;
document.head.appendChild(style);

updateStatus();