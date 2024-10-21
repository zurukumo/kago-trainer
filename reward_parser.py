import csv
import re
import os


class Parser():
    def __init__(self, years=[2015, 2016, 2017], max_count=1500000, debug=False):
        self.count = 0
        self.max_count = max_count
        self.debug = debug
        for year in years:
            file_dir = './xml' + str(year)
            for filename in os.listdir(file_dir):
                self.filename = filename
                self.ts = -1
                self.xml_parse(file_dir + '/' + filename)

    def url(self):
        return 'https://tenhou.net/0/?log={}&ts={}'.format(self.filename.replace('.xml', ''), self.ts)

    def xml_parse(self, filename):
        with open(filename, 'r') as xml:
            scs = []
            for elem, attr in re.findall(r'<(.*?)[ /](.*?)/?>', xml.read()):
                attr = dict(re.findall(r'\s?(.*?)="(.*?)"', attr))

                # 開局
                if elem == 'INIT':
                    kyoku, honba, kyotaku, _, _, _ = map(int, attr['seed'].split(','))
                    kyoku = min(kyoku, 99)
                    honba = min(honba, 99)
                    kyotaku = min(kyotaku, 99)

                elif elem == 'GO':
                    type = int(attr['type'])
                    # 人間
                    if not (type & 1) != 0:
                        break

                    # NOT 赤牌
                    if not (type & 2) == 0:
                        break

                    # なしなし
                    if not (type & 4) == 0:
                        break

                    # 東風
                    if not (type & 8) != 0:
                        break

                    # 三麻
                    if not (type & 16) == 0:
                        break

                # 和了
                elif elem == 'AGARI' or elem == 'RYUKYOKU':
                    sc = [int(i) for i in attr['sc'].split(',')]
                    print(sc)
                    nsc = []
                    for i in range(0, 8, 2):
                        s = (sc[i] + sc[i + 1]) // 5
                        s = min(s, 119)
                        s = max(s, 0)
                        nsc.append(s)
                    scs.append([kyoku, honba, kyotaku] + nsc)
                    print(self.url(), kyoku, honba, kyotaku)

        if scs:
            final_score = [scs[-1][i] for i in range(3, 7)]
            sorted_score = sorted(final_score)
            rank = [3 - sorted_score.index(final_score[i]) for i in range(4)]

            with open('reward.csv', 'a') as f:
                writer = csv.writer(f)
                for i in range(len(scs) - 1):
                    writer.writerow(rank + scs[i + 1][:3] + scs[i][3:])

            self.count += 1
            print(self.count, '/', self.max_count)
            if self.count == self.max_count:
                print('終了')
                exit()


Parser(
    years=[2015, 2016, 2017],
    max_count=1000000,
    debug=False
)
