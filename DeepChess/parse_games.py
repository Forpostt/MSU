import chess
import chess.pgn
import os


FIGURES = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # white
           'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}    # black
RESULT = {'1-0': 1, '0-1': 0, '1/2-1/2': -1}
MAX_GAMES_COUNT = 50000     # max games count for save if file


def games_in_file(path='./dataset/data.png'):
    f = open(path)
    g = chess.pgn.read_game(f)
    while g:
        yield g
        g = chess.pgn.read_game(f)


def board_to_vec(board, result_game):
    res = []
    board = str(board).split('\n')
    for i, row in enumerate(board):
        row = row.split()
        for j, cell in enumerate(row):
            if cell != '.':
                res.append(FIGURES[cell] * 64 + i * 8 + j)
    if result_game:
        res.append(768)
    return res   # positions of ones in vector, vector size 769


def extract_boards(path='./dataset/'):
    files = []
    for file in os.listdir(path):
        if file.endswith('.pgn'):
            files.append(path + file)
    win_position = []
    lose_position = []
    for file in files:
        for i, game in enumerate(games_in_file(file)):
            if i >= MAX_GAMES_COUNT:
                break
            if RESULT[game.headers['Result']] == -1:    # skip draw
                continue
            board = game.board()
            for move in game.main_line():
                board.push(move)
                if RESULT[game.headers['Result']]:
                    win_position.append(board_to_vec(board, 1))
                else:
                    lose_position.append(board_to_vec(board, 0))
    return win_position, lose_position


def dump_games(win_path='./data/win_games.txt', lose_path='./data/lose_games.txt', positions=(None, None)):
    if positions[0] is not None:
        f = open(win_path, 'w')
        for board in positions[0]:
            f.write(' '.join(map(str, board)) + '\n')
        f.close()
    if positions[1] is not None:
        f = open(lose_path, 'w')
        for board in positions[1]:
            f.write(' '.join(map(str, board)) + '\n')
        f.close()


if __name__ == '__main__':
    dump_games(positions=extract_boards())

