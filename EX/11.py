class Gomoku:
    def __init__(self, size=15):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = 'X'

    def display_board(self):
        for row in self.board:
            print(' | '.join(row))
            print('-' * (self.size * 4 - 3))

    def make_move(self, row, col):
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            if self.check_winner(row, col):
                print(f'Player {self.current_player} wins!')
                return True
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return False
        else:
            print('Invalid move. Try again.')
            return False

    def check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def play(self):
        while True:
            self.display_board()
            try:
                row = int(input(f'Player {self.current_player}, enter row (0-{self.size - 1}): '))
                col = int(input(f'Player {self.current_player}, enter column (0-{self.size - 1}): '))
                if self.make_move(row, col):
                    self.display_board()
                    break
            except ValueError:
                print('Invalid input. Please enter numbers only.')

if __name__ == '__main__':
    game = Gomoku()
    game.play()
