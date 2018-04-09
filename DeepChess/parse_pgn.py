import chess
import chess.pgn

MAX_GAMES_IN_FILES = 50000
FILES_COUNT = 4


def parse_pgn(path='./dataset/data.pgn'):
    f_in = open(path, 'r')
    count = 0
    while count < FILES_COUNT:
        f_out = open('./dataset/data_{}.pgn'.format(count), 'w')
        for i in range(MAX_GAMES_IN_FILES):
            game = chess.pgn.read_game(f_in)
            if not game:
                f_out.close()
                return
            f_out.write(str(game) + '\n\n')
        f_out.close()
        count += 1
        print count


if __name__ == '__main__':
    parse_pgn()
