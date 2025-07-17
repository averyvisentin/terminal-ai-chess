# main.py
import blessed
import chess
import chess.pgn
import ollama
import sys
import os
import traceback

class GameLauncher:
    """
    Handles the pre-game menus, including the main menu and settings screen.
    """
    def __init__(self, term):
        self.term = term
        self.difficulties = ["Rookie", "Novice", "Normal", "Expert", "Grandmaster"]
        self.settings = {
            "model_name": "gemma3:latest",
            "temperature": 0.1,
            "difficulty": "Normal",
        }

    def _draw_menu(self, title, options, selected_index):
        """A generic function to draw a menu with a title and selectable options."""
        print(self.term.home + self.term.clear)
        # Center the title
        title_x = (self.term.width - len(title)) // 2
        print(self.term.move_xy(title_x, 2) + self.term.bold(title))

        for i, option in enumerate(options):
            option_x = (self.term.width - len(option)) // 2
            option_y = 5 + i * 2
            if i == selected_index:
                print(self.term.move_xy(option_x - 2, option_y) + self.term.black_on_white(f"> {option} <"))
            else:
                print(self.term.move_xy(option_x, option_y) + option)
        sys.stdout.flush()

    def run_main_menu(self):
        """Displays and handles the main menu navigation."""
        options = ["Start New Game", "Settings", "Quit"]
        selected_index = 0
        with self.term.cbreak(), self.term.hidden_cursor():
            while True:
                self._draw_menu("Terminal Chess LLM", options, selected_index)
                key = self.term.inkey()
                if key.is_sequence:
                    if key.code == self.term.KEY_UP:
                        selected_index = (selected_index - 1) % len(options)
                    elif key.code == self.term.KEY_DOWN:
                        selected_index = (selected_index + 1) % len(options)
                    elif key.code == self.term.KEY_ENTER:
                        return options[selected_index]
                elif key.lower() == 'q':
                    return "Quit"

    def run_settings_menu(self):
        """Displays and handles the settings menu."""
        while True:
            options = [
                f"AI Model: {self.settings['model_name']}",
                f"Temperature: {self.settings['temperature']:.1f}",
                f"Difficulty: < {self.settings['difficulty']} >",
                "Back to Main Menu"
            ]
            title = "Settings"
            selected_index = 0

            with self.term.cbreak(), self.term.hidden_cursor():
                while True:
                    self._draw_menu(title, options, selected_index)
                    key = self.term.inkey()

                    if key.is_sequence:
                        if key.code == self.term.KEY_UP:
                            selected_index = (selected_index - 1) % len(options)
                        elif key.code == self.term.KEY_DOWN:
                            selected_index = (selected_index + 1) % len(options)
                        elif key.code == self.term.KEY_ENTER:
                            if selected_index == 0: # Change Model
                                self._edit_setting_string("AI Model", "model_name")
                                break
                            elif selected_index == 1: # Change Temperature
                                self._edit_setting_float("Temperature", "temperature")
                                break
                            elif selected_index == 2: # Change Difficulty
                                self._cycle_difficulty()
                                break
                            elif selected_index == 3: # Back
                                return
                    elif key.lower() == 'q':
                        return

    def _cycle_difficulty(self):
        """Cycles through the available difficulty levels."""
        current_index = self.difficulties.index(self.settings['difficulty'])
        next_index = (current_index + 1) % len(self.difficulties)
        self.settings['difficulty'] = self.difficulties[next_index]

    def _edit_setting_string(self, prompt, setting_key):
        """Helper to edit a string-based setting."""
        print(self.term.move_xy(0, 10) + self.term.clear_eol())
        current_val = self.settings[setting_key]
        with self.term.hidden_cursor(False):
             new_val = self.term.prompt(f"Enter new {prompt} (current: {current_val}): ")
        if new_val:
            self.settings[setting_key] = new_val

    def _edit_setting_float(self, prompt, setting_key):
        """Helper to edit a float-based setting."""
        print(self.term.move_xy(0, 10) + self.term.clear_eol())
        current_val = self.settings[setting_key]
        with self.term.hidden_cursor(False):
             new_val_str = self.term.prompt(f"Enter new {prompt} (current: {current_val}): ")
        try:
            if new_val_str:
                self.settings[setting_key] = float(new_val_str)
        except ValueError:
            pass # Ignore invalid float input


class ChessGame:
    """
    The main class that orchestrates a terminal-based chess game between a
    human player (White) and an LLM opponent (Black).
    """

    def __init__(self, term, model_name="gemma3:latest", temperature=0.1, difficulty="Normal"):
        """
        Initializes all game components: the terminal interface, the chess board
        engine, and the connection to the Ollama LLM client.

        Args:
            model_name (str): The name of the Ollama model to use as the opponent.
            difficulty (str): The skill level of the AI opponent.
        """
        # --- Core Components ---
        self.term = term
        self.board = chess.Board()
        self.model_name = model_name
        self.temperature = temperature
        self.difficulty = difficulty
        self.ollama_client = None  # Initialized in self.connect_to_ollama()

        # --- UI Configuration ---
        self.SQUARE_WIDTH = 4
        self.SQUARE_HEIGHT = 2
        self.BOARD_ORIGIN_Y = 2
        self.BOARD_ORIGIN_X = 4

        # --- Nerd Font Piece Mapping ---
        # Note: A Nerd Font must be installed and used in the terminal
        # for these icons to render correctly.
        self.NERD_FONT_PIECES = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        }

        # --- Dynamic Color Selection based on Terminal Capabilities ---
        if self.term.number_of_colors >= 256:
            self.light_square_bg = self.term.on_color(222)
            self.dark_square_bg = self.term.on_color(130)
            self.cursor_color = self.term.on_yellow
            self.selected_color = self.term.on_green
            self.legal_move_color = self.term.on_bright_blue
            self.white_piece_fg = self.term.black
            self.black_piece_fg = self.term.white
        else:
            self.light_square_bg = self.term.on_white
            self.dark_square_bg = self.term.on_bright_black
            self.cursor_color = self.term.on_yellow
            self.selected_color = self.term.on_green
            self.legal_move_color = self.term.on_blue
            self.white_piece_fg = self.term.black
            self.black_piece_fg = self.term.white

        # --- Game State ---
        self.game_running = True
        self.status_message = "Welcome! Use arrow keys to move, Enter to select."
        self.move_history = []
        self._last_width = 0
        self._last_height = 0

        # --- Cursor and Selection State ---
        self.cursor_row = 7  # Board is 0-7, rank 1 is row 7
        self.cursor_col = 4  # File 'e'
        self.selected_square = None

    def connect_to_ollama(self):
        """
        Establishes and verifies the connection to the Ollama server.
        If the model is not found locally, it attempts to pull it.
        Exits the application if the connection or pull fails.
        """
        try:
            self.ollama_client = ollama.Client()
            # Check if the model exists locally with robust parsing
            response = self.ollama_client.list()

            if 'models' not in response or not isinstance(response['models'], list):
                 print(self.term.red("Error: Unexpected response format from Ollama server."))
                 print("Could not find 'models' list in the response.")
                 return False

            available_models = [
                model_info['name']
                for model_info in response.get('models', [])
                if isinstance(model_info, dict) and 'name' in model_info
            ]

            if self.model_name not in available_models:
                print(self.term.yellow(f"Model '{self.model_name}' not found locally."))
                print(self.term.yellow(f"Attempting to pull the model from Ollama. This may take a while..."))
                try:
                    # Use a streaming pull to provide feedback to the user.
                    last_status = ""
                    for progress in self.ollama_client.pull(self.model_name, stream=True):
                        status = progress.get('status')
                        if status and status != last_status:
                            # Erase the progress bar line before printing a new status
                            print(self.term.move_up(1) + self.term.clear_eol() if 'total' in progress else '')
                            print(f"  {status}")
                            last_status = status

                        # A simple progress indicator for downloads
                        if 'total' in progress and 'completed' in progress:
                            total = progress['total']
                            completed = progress['completed']
                            if total > 0:
                                percent = round((completed / total) * 100)
                                bar_length = 30
                                filled_length = int(bar_length * completed // total)
                                bar = '█' * filled_length + '─' * (bar_length - filled_length)
                                # Use carriage return to show progress on a single line
                                print(f"  Downloading: [{bar}] {percent}%", end='\r')

                    print("\n" + self.term.green(f"Successfully pulled model '{self.model_name}'."))

                except ollama.ResponseError as pull_error:
                    print(self.term.red(f"\nError: Failed to pull model '{self.model_name}'."))
                    print(self.term.red(f"Ollama API Error: {pull_error.error}"))
                    print("Please check that the model name is correct and that you are connected to the internet.")
                    return False
                except Exception as pull_error:
                    print(self.term.red(f"\nAn unexpected error occurred while pulling the model."))
                    print(f"Details: {pull_error.__class__.__name__}: {pull_error}")
                    return False

        except ollama.RequestError as e:
            print(self.term.red("Error: Could not connect to Ollama. The server may not be running or is unreachable."))
            print(f"Details: {e.__class__.__name__}: {e}")
            return False
        except Exception as e:
            print(self.term.red("An unexpected error occurred during setup."))
            print(f"Details: {e.__class__.__name__}: {e}")
            return False
        return True # Connection successful

    # ==============================================================================
    # TUI Drawing Methods
    # ==============================================================================

    def _draw_full_ui(self):
        """Clears the screen and redraws the entire user interface."""
        print(self.term.home + self.term.clear, end='')
        self._draw_board_grid()
        self._draw_legal_moves() # Draw hints before pieces
        self._draw_pieces()
        self._draw_cursor() # Draw cursor on top of pieces
        self._draw_coordinates()
        self._draw_move_history()
        self._draw_status_bar()
        sys.stdout.flush()

    def _draw_board_grid(self):
        """Draws the 8x8 checkered grid for the chessboard."""
        for row in range(8):
            for col in range(8):
                is_light_square = (row + col) % 2 == 0
                bg_color = self.light_square_bg if is_light_square else self.dark_square_bg

                # Highlight the selected square
                current_square = chess.square(col, 7 - row)
                if self.selected_square == current_square:
                    bg_color = self.selected_color

                screen_y = self.BOARD_ORIGIN_Y + row * self.SQUARE_HEIGHT
                screen_x = self.BOARD_ORIGIN_X + col * self.SQUARE_WIDTH

                for r_offset in range(self.SQUARE_HEIGHT):
                    with self.term.location(x=screen_x, y=screen_y + r_offset):
                        print(bg_color + ' ' * self.SQUARE_WIDTH, end='')

    def _draw_cursor(self):
        """Draws a border around the square currently under the cursor."""
        screen_y = self.BOARD_ORIGIN_Y + self.cursor_row * self.SQUARE_HEIGHT
        screen_x = self.BOARD_ORIGIN_X + self.cursor_col * self.SQUARE_WIDTH

        # Use a bright color for the cursor border
        cursor_border_color = self.term.yellow

        # Draw top and bottom borders
        with self.term.location(x=screen_x, y=screen_y):
            print(cursor_border_color + '─' * self.SQUARE_WIDTH, end='')
        with self.term.location(x=screen_x, y=screen_y + self.SQUARE_HEIGHT -1):
            print(cursor_border_color + '─' * self.SQUARE_WIDTH, end='')

        # Draw side borders
        for r_offset in range(self.SQUARE_HEIGHT):
            with self.term.location(x=screen_x, y=screen_y + r_offset):
                print(cursor_border_color + '│', end='')
            with self.term.location(x=screen_x + self.SQUARE_WIDTH - 1, y=screen_y + r_offset):
                print(cursor_border_color + '│', end='')

    def _draw_legal_moves(self):
        """Draws a 'ghost path' or indicator on squares that are legal moves."""
        if self.selected_square is None:
            return

        for move in self.board.legal_moves:
            if move.from_square == self.selected_square:
                to_row = 7 - chess.square_rank(move.to_square)
                to_col = chess.square_file(move.to_square)

                screen_y = self.BOARD_ORIGIN_Y + to_row * self.SQUARE_HEIGHT + (self.SQUARE_HEIGHT // 2)
                screen_x = self.BOARD_ORIGIN_X + to_col * self.SQUARE_WIDTH + (self.SQUARE_WIDTH // 2)

                # Determine background color of the target square
                is_light_square = (to_row + to_col) % 2 == 0
                bg_color = self.light_square_bg if is_light_square else self.dark_square_bg

                with self.term.location(x=screen_x, y=screen_y):
                    # Draw a dot as the legal move indicator
                    print(bg_color + self.term.black + '●' + self.term.normal, end='')


    def _draw_pieces(self):
        """
        Renders the pieces on the board based on the canonical state from the
        `chess.Board` object. This version renders a single, centered piece icon.
        """
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Use the Nerd Font mapping, falling back to the piece's symbol if not found
                symbol = self.NERD_FONT_PIECES.get(piece.symbol(), piece.symbol())
                piece_color = self.white_piece_fg if piece.color == chess.WHITE else self.black_piece_fg

                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)

                # Center the single character within the square
                screen_y = self.BOARD_ORIGIN_Y + row * self.SQUARE_HEIGHT + (self.SQUARE_HEIGHT // 2)
                screen_x = self.BOARD_ORIGIN_X + col * self.SQUARE_WIDTH + (self.SQUARE_WIDTH // 2)

                is_light_square = (row + col) % 2 == 0
                bg_color = self.light_square_bg if is_light_square else self.dark_square_bg

                if self.selected_square == square:
                    bg_color = self.selected_color

                with self.term.location(x=screen_x, y=screen_y):
                    print(bg_color + piece_color + symbol + self.term.normal, end='')

    def _draw_coordinates(self):
        """Draws the file (a-h) and rank (1-8) labels around the board."""
        for col in range(8):
            screen_x = self.BOARD_ORIGIN_X + col * self.SQUARE_WIDTH + (self.SQUARE_WIDTH // 2)
            screen_y = self.BOARD_ORIGIN_Y + 8 * self.SQUARE_HEIGHT
            with self.term.location(x=screen_x, y=screen_y):
                print(self.term.bold(chr(ord('a') + col)), end='')

        for row in range(8):
            screen_y = self.BOARD_ORIGIN_Y + row * self.SQUARE_HEIGHT + (self.SQUARE_HEIGHT // 2)
            screen_x = self.BOARD_ORIGIN_X - 2
            with self.term.location(x=screen_x, y=screen_y):
                print(self.term.bold(str(8 - row)), end='')

    def _draw_move_history(self):
        """Displays the move history as a chat log."""
        history_x = self.BOARD_ORIGIN_X + 8 * self.SQUARE_WIDTH + 5
        history_y = self.BOARD_ORIGIN_Y
        with self.term.location(x=history_x, y=history_y):
            print(self.term.bold("Move History"))

        max_history_rows = self.term.height - history_y - 2

        # Display the most recent moves that fit in the allotted space
        start_index = max(0, len(self.move_history) - max_history_rows)
        display_moves = self.move_history[start_index:]

        for i, log_entry in enumerate(display_moves):
            player = log_entry['player']
            piece_name = log_entry['piece_name']
            from_square = log_entry['from_square']
            to_square = log_entry['to_square']

            log_line = f"{player}: {piece_name} {from_square} to {to_square}"

            if log_entry['capture']:
                captured_piece_name = chess.piece_name(log_entry['captured_piece_type']).capitalize()
                log_line += f" (takes {captured_piece_name})"

            with self.term.location(x=history_x, y=history_y + 2 + i):
                print(self.term.clear_eol + log_line)

    def _draw_status_bar(self):
        """
        Displays the current status message at the bottom of the terminal.
        """
        status_y = self.BOARD_ORIGIN_Y + 8 * self.SQUARE_HEIGHT + 2
        with self.term.location(x=0, y=status_y):
            print(self.term.clear_eol + self.term.bold(self.status_message))

    # ==============================================================================
    # Game Logic and Input Handling
    # ==============================================================================

    def _get_user_input(self):
        """
        Handles cursor-based keyboard input for the human player's move.
        Returns a valid `chess.Move` object or None if the user quits.
        """
        while self.game_running:
            self._draw_full_ui()
            key = self.term.inkey(timeout=5) # Use a long timeout

            if key.is_sequence:
                if key.code == self.term.KEY_UP:
                    self.cursor_row = max(0, self.cursor_row - 1)
                elif key.code == self.term.KEY_DOWN:
                    self.cursor_row = min(7, self.cursor_row + 1)
                elif key.code == self.term.KEY_LEFT:
                    self.cursor_col = max(0, self.cursor_col - 1)
                elif key.code == self.term.KEY_RIGHT:
                    self.cursor_col = min(7, self.cursor_col + 1)
                elif key.code == self.term.KEY_ESCAPE:
                    self.selected_square = None
                    self.status_message = "Selection cancelled."
                elif key.code == self.term.KEY_ENTER:
                    cursor_square = chess.square(self.cursor_col, 7 - self.cursor_row)

                    if self.selected_square is not None:
                        # A piece is already selected, try to make a move
                        move = chess.Move(self.selected_square, cursor_square)
                        # Check for promotion
                        piece = self.board.piece_at(self.selected_square)
                        if piece.piece_type == chess.PAWN and chess.square_rank(cursor_square) in [0, 7]:
                            move.promotion = chess.QUEEN # Auto-promote to Queen for simplicity

                        if move in self.board.legal_moves:
                            self.selected_square = None
                            return move # Valid move confirmed
                        else:
                            self.status_message = "Invalid move. Selection cancelled."
                            self.selected_square = None
                    else:
                        # No piece is selected, try to select one
                        piece = self.board.piece_at(cursor_square)
                        if piece and piece.color == self.board.turn:
                            self.selected_square = cursor_square
                            self.status_message = f"Selected {self.NERD_FONT_PIECES.get(piece.symbol())} at {chess.square_name(cursor_square)}. Choose destination."
                        else:
                            self.status_message = "Cannot select an empty square or opponent's piece."
            elif key.lower() == 'q':
                return None # Signal to quit

    def _get_llm_move(self):
        """
        Generates the LLM's move using a robust, selective retry loop.
        """
        max_retries = 30

        # Define difficulty personas
        difficulty_personas = {
            "Rookie": "You are a chess rookie. You make mistakes and blunders frequently.",
            "Novice": "You are a novice chess player. You understand the rules but lack deep strategy.",
            "Normal": "You are a skilled club-level chess player.",
            "Expert": "You are a chess expert. You play with deep strategic and tactical understanding.",
            "Grandmaster": "You are a world-class chess grandmaster. You play with the utmost precision and find the best possible move in any position, even difficult ones. Analyze the full game history to anticipate your opponent's plan and formulate a long-term strategy."
        }

        # Select the persona based on the current difficulty setting
        persona = difficulty_personas.get(self.difficulty, "You are a chess AI playing as Black.")

        # Generate the list of legal moves once for this turn.
        legal_moves = list(self.board.legal_moves)

        # Create a formatted string of legal moves for the prompt.
        prompt_moves = "\n".join([f"{i+1}. {self.board.san(move)}" for i, move in enumerate(legal_moves)])

        # The PGN is now included for the Grandmaster difficulty to provide context.
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        pgn_string = game.accept(exporter)

        prompt = f"""{persona}
The current board state is: {self.board.fen()}
The game history so far is: {pgn_string}

Here is a list of your legal moves:
{prompt_moves}

Your task is to choose the single best move from this list according to your skill level.
Respond with ONLY the number corresponding to your chosen move. For example, if you choose the first move, respond with '1'.
"""

        for attempt in range(max_retries):
            self.status_message = f"Black ({self.model_name}) is thinking..."
            if attempt > 0:
                self.status_message = f"LLM provided an invalid choice. Retrying ({attempt+1}/{max_retries})..."
            self._draw_full_ui()

            try:
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': self.temperature}
                )

                response_text = response['message']['content'].strip()

                # Attempt to parse the response as an integer.
                choice_index = int(response_text) - 1

                # Validate the choice.
                if 0 <= choice_index < len(legal_moves):
                    return legal_moves[choice_index] # Success!
                else:
                    # The number was out of range.
                    self.status_message = f"LLM chose an invalid number: {response_text}. Retrying..."
                    self._draw_full_ui()
                    self.term.inkey(timeout=1)

            except (ValueError, IndexError):
                # The response was not a valid number or was out of bounds.
                self.status_message = f"LLM response was not a valid choice: '{response_text[:20]}...'. Retrying..."
                self._draw_full_ui()
                self.term.inkey(timeout=1)
            except Exception as e:
                self.status_message = f"An error occurred with Ollama: {e}"
                self._draw_full_ui()
                return None

        self.status_message = "LLM failed to provide a valid move after multiple attempts."
        self._draw_full_ui()
        return None

    def _create_log_entry(self, move):
        """
        Gathers all information about a move *before* it is pushed to the board.
        Returns a dictionary representing the log entry.
        """
        player = "White" if self.board.turn == chess.WHITE else "Black"
        is_capture = self.board.is_capture(move)
        captured_piece_type = None

        moving_piece = self.board.piece_at(move.from_square)
        piece_name = chess.piece_name(moving_piece.piece_type).capitalize()
        from_square_name = chess.square_name(move.from_square)
        to_square_name = chess.square_name(move.to_square)

        if is_capture:
            if self.board.is_en_passant(move):
                captured_piece_type = chess.PAWN
            else:
                captured_piece = self.board.piece_at(move.to_square)
                if captured_piece:
                    captured_piece_type = captured_piece.piece_type

        # Get the SAN before pushing the move to the board.
        san = self.board.san(move)

        return {
            'player': player,
            'piece_name': piece_name,
            'from_square': from_square_name,
            'to_square': to_square_name,
            'capture': is_capture,
            'captured_piece_type': captured_piece_type,
            'san': san
        }

    def run(self):
        """The main game loop."""
        MIN_HEIGHT = self.BOARD_ORIGIN_Y + 8 * self.SQUARE_HEIGHT + 3
        MIN_WIDTH = self.BOARD_ORIGIN_X + 8 * self.SQUARE_WIDTH + 20
        if self.term.height < MIN_HEIGHT or self.term.width < MIN_WIDTH:
            print(self.term.red("Error: Terminal window is too small."))
            print(f"Required dimensions: > {MIN_WIDTH} columns, > {MIN_HEIGHT} rows.")
            self.term.inkey()
            return

        if not self.connect_to_ollama():
            self.term.inkey()
            return

        self._last_width = self.term.width
        self._last_height = self.term.height

        with self.term.cbreak(), self.term.hidden_cursor():
            while self.game_running and not self.board.is_game_over(claim_draw=True):
                if self.board.turn == chess.WHITE:
                    move = self._get_user_input()
                    if move is None:
                        self.game_running = False
                        self.status_message = "Game aborted by user."
                        continue

                    log_entry = self._create_log_entry(move)
                    self.board.push(move)
                    self.move_history.append(log_entry)
                    self.status_message = f"You played {log_entry['san']}. Black is thinking..."
                else: # LLM's turn
                    move = self._get_llm_move()
                    if move:
                        log_entry = self._create_log_entry(move)
                        self.board.push(move)
                        self.move_history.append(log_entry)
                        self.status_message = f"Black played {log_entry['san']}. Your turn."
                    else:
                        self.game_running = False

        print(self.term.home + self.term.clear)
        print(self.term.bold("Game Over\n"))

        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            result_text = outcome.result()
            termination_reason = outcome.termination.name.replace('_', ' ').title()
            print(f"Result: {result_text} - {termination_reason}")
        else:
            print(self.status_message)

        print("\nFinal Board:")
        print(self.board.unicode(borders=True, empty_square='·'))

        print("\nPGN:")
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn_string = game.accept(exporter)
        print(pgn_string)

        print("\nPress any key to exit.")
        self.term.inkey()

def main():
    """The main entry point for the application."""
    term = blessed.Terminal()
    launcher = GameLauncher(term)

    while True:
        choice = launcher.run_main_menu()
        if choice == "Start New Game":
            game = ChessGame(term, **launcher.settings)
            game.run()
        elif choice == "Settings":
            launcher.run_settings_menu()
        elif choice == "Quit":
            break

    print(term.home + term.clear)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame interrupted. Exiting.")
    except Exception as e:
        print("An unexpected error occurred. Please see traceback below.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        term = blessed.Terminal()
        print(term.normal)
        print("Terminal restored. Goodbye!")
