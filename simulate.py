from time import sleep, time

from server.game import Game

game = Game(debug=True)
game.start_game()


def draw_board():
    H = W = 24
    board = [[' '] * W for _ in range(H)]

    tehai_start = ((H - 1, 1), (H - 2, W - 1), (0, W - 2), (1, 0))
    kawa_start = ((H - 5, W // 2 - 2), (H // 2 + 2, W - 5), (5, W // 2 + 1), (H // 2 - 1, 5))
    move = ((0, 1), (-1, 0), (0, -1), (1, 0))
    shift = ((1, -6), (6, 1), (-1, 6), (-6, -1))
    color = (
        '\033[31m{}\033[0m',
        '\033[37m{}\033[0m',
        '\033[32m{}\033[0m',
        '\033[36m{}\033[0m',
    )

    print('{}{}局\n {} {} {} {}'.format(
        '東南西北'[game.kyoku // 4],
        game.kyoku % 4 + 1,
        game.scores[0],
        game.scores[1],
        game.scores[2],
        game.scores[3],
    ))

    for i, player in enumerate(game.players):
        if not hasattr(player, 'tehai'):
            continue

        y, x = tehai_start[i]
        dy, dx = move[i]

        for p in player.tehai:
            p = p // 4
            board[y][x] = color[p // 9].format(p % 9 + 1)
            y += dy
            x += dx

        for huro in player.huro:
            y += dy
            x += dx
            for p in huro['pais']:
                p = p // 4
                board[y][x] = color[p // 9].format(p % 9 + 1)
                y += dy
                x += dx

        y, x = kawa_start[i]
        sy, sx = shift[i]
        for j, p in enumerate(player.kawa):
            p = p // 4
            board[y][x] = color[p // 9].format(p % 9 + 1)
            y += dy
            x += dx
            if j % 6 == 5:
                y += sy
                x += sx

    for b in board:
        print(*b)
    print("=" * W)
    sleep(0.2)


p = time()
while True:
    # state = game.state
    game.next()
    # c = time()
    # print("{}\t{}".format(state, c - p))
    # p = c
    # draw_board()
    # input()
