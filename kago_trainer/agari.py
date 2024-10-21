# !/usr/bin/python
# coding: UTF-8

from itertools import combinations


class Agari():
    # 定数
    YAOCHU1 = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    YAOCHU2 = [0, 6, 9, 15, 18, 24]
    YAKU = [
        '門前清自摸和',
        '立直',
        '一発',
        '槍槓',
        '嶺上開花',
        '海底摸月',
        '河底撈魚',
        '平和',
        '断幺九',
        '一盃口',
        '自風 東',
        '自風 南',
        '自風 西',
        '自風 北',
        '場風 東',
        '場風 南',
        '場風 西',
        '場風 北',
        '役牌 白',
        '役牌 發',
        '役牌 中',
        '両立直',
        '七対子',
        '混全帯幺九',
        '一気通貫',
        '三色同順',
        '三色同刻',
        '三槓子',
        '対々和',
        '三暗刻',
        '小三元',
        '混老頭',
        '二盃口',
        '純全帯幺九',
        '混一色',
        '清一色',
        '',
        '天和',
        '地和',
        '大三元',
        '四暗刻',
        '四暗刻単騎',
        '字一色',
        '緑一色',
        '清老頭',
        '九蓮宝燈',
        '純正九蓮宝燈',
        '国士無双',
        '国士無双１３面',
        '大四喜',
        '小四喜',
        '四槓子',
        'ドラ',
        '裏ドラ',
        '赤ドラ']

    def __init__(self, jun, huro, ba, jokyo_yaku):
        self.jun = jun
        self.huro = huro

        # ba = [kyoku, honba, kyotaku, who, fromWho, (paoWho)]
        self.ba = ba

        self.all_agari = []

        # 手牌のビットマップ作成
        self.map = jun.copy()

        for i in range(0, len(huro), 3):
            if huro[i + 2] <= -2:
                self.map[huro[i]] += 3
            else:
                self.map[huro[i]] += 1
                self.map[huro[i] + 1] += 1
                self.map[huro[i] + 2] += 1

        self.menzen = 1 if len(huro) - huro.count(-16) * 3 == 0 else 0

        self.tsumo = 1 if ba[3] == ba[4] else 0

        # 状況役と全部役を定義
        self.jokyo_yaku = jokyo_yaku
        self.zenbu_yaku = self.get_zenbu_yaku()

        # 一般手の和了形を全て抜き出す
        for i in range(34):
            if self.jun[i] >= 2:
                self.jun[i] -= 2
                self.get_mentsu([], i, 0)
                self.jun[i] += 2

    def get_ten(self, machi):
        max = [0, 0]
        for agari in self.all_agari:
            for i in range(len(self.huro), 13):
                if machi == agari[i]:
                    # 双ポンロンの場合明刻に変換
                    if not self.tsumo and i in range(0, 3 * 4, 3) and agari[i + 2] == -4:
                        agari[i + 2] = -2
                        hu = self.get_hu(agari, i)
                        bubun_yaku = self.get_bubun_yaku(agari)
                        agari[i + 2] = -4
                    else:
                        hu = self.get_hu(agari, i)
                        bubun_yaku = self.get_bubun_yaku(agari)

                    # 平和
                    if hu == 20 and self.menzen:
                        bubun_yaku[7] = 1

                    han = sum(self.jokyo_yaku) + sum(self.zenbu_yaku) + sum(bubun_yaku)

                    # ツモ符計算
                    # 喰い平和形
                    if hu == 20 and not self.menzen:
                        hu = 30
                    # 門前ロン
                    elif (not self.tsumo) and self.menzen:
                        hu += 10
                    # ツモ(平和除く)
                    elif hu != 20 and self.tsumo:
                        hu += 2

                    hu = -(-hu // 10) * 10

                    if hu * (2 ** han) > max[0] * (2 ** max[1]):
                        max = [hu, han]
                        self.bubun_yaku = bubun_yaku

        # 七対子
        if self.jun.count(2) == 7:
            han = sum(self.jokyo_yaku) + sum(self.zenbu_yaku) + 2

            if 25 * (2 ** han) > max[0] * (2 ** max[1]):
                max = [25, han]
                self.bubun_yaku = [2 if i == 22 else 0 for i in range(0, 55)]

        # 国士無双
        if sum([1 if self.jun[i] >= 1 else -1 for i in Agari.YAOCHU1]) == 13:
            han = sum(self.jokyo_yaku[37:39]) + 13
            max = [1, han]
            self.bubun_yaku = [13 if i == 47 else 0 for i in range(0, 55)]

        if sum(self.jokyo_yaku[37:52] + self.zenbu_yaku[37:52] + self.bubun_yaku[37:52]) != 0:
            self.jokyo_yaku = [self.jokyo_yaku[i] if 37 <= i <= 51 else 0 for i in range(0, 55)]
            self.zenbu_yaku = [self.zenbu_yaku[i] if 37 <= i <= 51 else 0 for i in range(0, 55)]
            self.bubun_yaku = [self.bubun_yaku[i] if 37 <= i <= 51 else 0 for i in range(0, 55)]
            han = sum(self.jokyo_yaku + self.zenbu_yaku + self.bubun_yaku)

        # 点数変動計算
        ten_exchange = [0, 0, 0, 0]

        ten = max[0] * (2 ** (max[1] + 2))

        if ten >= 2000:
            if max[1] >= 13:
                ten = 8000 * (max[1] // 13)
            elif max[1] >= 11:
                ten = 6000
            elif max[1] >= 8:
                ten = 4000
            elif max[1] >= 6:
                ten = 3000
            else:
                ten = 2000

        if self.ba[3] == self.ba[0] % 4:
            ten *= 6
        else:
            ten *= 4

        ten = -(-ten // 100)

        if self.tsumo:
            # 包ツモ
            if len(self.ba) == 6:
                ten_exchange[self.ba[3]] = ten + self.ba[1] * 3 + self.ba[2] * 10
                ten_exchange[self.ba[5]] = -ten - self.ba[1] * 3

            # 親ツモ
            elif self.ba[3] == self.ba[0] % 4:
                for i in range(0, 4):
                    if i == self.ba[3]:
                        ten_exchange[i] = -(-ten // 3) * 3 + (self.ba[1] * 3) + (self.ba[2] * 10)
                    else:
                        ten_exchange[i] = -ten // 3 - self.ba[1]
            # 子ツモ
            else:
                for i in range(0, 4):
                    if i == self.ba[3]:
                        ten_exchange[i] += self.ba[2] * 10
                    elif i == self.ba[0] % 4:
                        ten_exchange[i] = -ten // 2 - self.ba[1]
                        ten_exchange[self.ba[3]] -= ten_exchange[i]
                    else:
                        ten_exchange[i] = -ten // 4 - self.ba[1]
                        ten_exchange[self.ba[3]] -= ten_exchange[i]

        else:
            # 包ロン(包者と放銃者が存在)
            if len(self.ba) == 6:
                ten_exchange[self.ba[3]] = ten + self.ba[1] * 3 + self.ba[2] * 10
                ten_exchange[self.ba[4]] = -ten // 2
                ten_exchange[self.ba[5]] = -ten // 2 - self.ba[1] * 3

            # ロン
            else:
                ten_exchange[self.ba[3]] = ten + self.ba[1] * 3 + self.ba[2] * 10
                ten_exchange[self.ba[4]] = -ten - self.ba[1] * 3

        return [i * 100 for i in ten_exchange]

    def get_simple_ten(self):
        max = 0

        for agari in self.all_agari:
            bubun_yaku = self.get_bubun_yaku(agari)

            if sum(bubun_yaku) + sum(self.zenbu_yaku) + sum(self.jokyo_yaku) > max:
                max = sum(bubun_yaku) + sum(self.zenbu_yaku) + sum(self.jokyo_yaku)
                self.bubun_yaku = bubun_yaku

        if sum(self.jokyo_yaku[37:52] + self.zenbu_yaku[37:52] + self.bubun_yaku[37:52]) != 0:
            self.jokyo_yaku = [self.jokyo_yaku[i] if 37 <= i <= 51 else 0 for i in range(55)]
            self.zenbu_yaku = [self.zenbu_yaku[i] if 37 <= i <= 51 else 0 for i in range(55)]
            self.bubun_yaku = [self.bubun_yaku[i] if 37 <= i <= 51 else 0 for i in range(55)]

        ten = 30 * (2 ** (max + 2))

        if ten >= 2000:
            if max >= 13:
                ten = 8000 * (max // 13)
            elif max >= 11:
                ten = 6000
            elif max >= 8:
                ten = 4000
            elif max >= 6:
                ten = 3000
            else:
                ten = 2000

        if self.ba[3] == self.ba[0] % 4:
            ten *= 6
        else:
            ten *= 4

        ten = -(-ten // 100) * 100

        return ten

    # 全面子抜き出し
    def get_mentsu(self, jun, janto, start):
        if sum(self.jun) == 0:
            self.all_agari.append(self.huro + jun + [janto, janto])
            return

        if sum(self.jun[0:start]) != 0:
            return

        for i in range(start, 34):
            if self.jun[i] == 0:
                continue

            # 順子抜き出し
            if i <= 26 and i % 9 <= 6 and self.jun[i + 1] and self.jun[i + 2]:
                self.jun[i] -= 1
                self.jun[i + 1] -= 1
                self.jun[i + 2] -= 1
                self.get_mentsu(jun + [i, i + 1, i + 2], janto, i)
                self.jun[i] += 1
                self.jun[i + 1] += 1
                self.jun[i + 2] += 1

            # 刻子抜き出し
            if self.jun[i] >= 3:
                self.jun[i] -= 3
                self.get_mentsu(jun + [i, -1, -4], janto, i)
                self.jun[i] += 3

    # ツモ符を除く符計算を行う
    def get_hu(self, agari, i):
        # 基本符
        hu = 20

        # 面子による符
        for j in range(0, 3 * 4, 3):
            tmp_hu = 0

            if agari[j + 2] <= -2:
                tmp_hu = agari[j + 2] * -1

            if agari[j] in Agari.YAOCHU1:
                tmp_hu *= 2

            hu += tmp_hu

        # 雀頭による符
        # 自風牌
        if agari[12] == (self.ba[3] - self.ba[0]) % 4 + 27:
            hu += 2
        # 場風牌
        if agari[12] == self.ba[0] // 4 + 27:
            hu += 2
        # 三元牌
        if 31 <= agari[12] <= 33:
            hu += 2

        # 待ちによる符
        # 単騎待ち
        if i == 12:
            hu += 2
        # 嵌張待ち
        elif i % 3 == 1 and agari[i + 1] > 0:
            hu += 2
        # 辺張待ち(123)
        elif i % 3 == 2 and agari[i - 2] % 9 == 0 and agari[i] > 0:
            hu += 2
        # 辺張待ち(789)
        elif i % 3 == 0 and agari[i] % 9 == 6 and agari[i + 2] > 0:
            hu += 2

        return hu

    # 全部役
    def get_zenbu_yaku(self):
        zenbu_yaku = [0 for i in range(0, 55)]

        # 役満
        # 39:大三元
        if self.map[31] + self.map[32] + self.map[33] == 9:
            zenbu_yaku[39] = 13

        # 42:字一色
        if sum(self.map[27:34]) == 14:
            zenbu_yaku[42] = 13

        # 43:緑一色
        if sum([self.map[i] for i in [19, 20, 21, 23, 25, 32]]) == 14:
            zenbu_yaku[43] = 13

        # 44:清老頭
        if sum([self.map[i] for i in [0, 8, 9, 17, 18, 26]]) == 14:
            zenbu_yaku[44] = 13

        # 45:九蓮宝燈
        if self.huro == []:
            for i in [0, 9, 18]:
                if self.map[i] >= 3 and self.map[i + 8] >= 3 and 0 not in self.map[i + 1:i + 8]:
                    zenbu_yaku[45] = 13

        # 49:大四喜
        if self.map[27] + self.map[28] + self.map[29] + self.map[30] == 12:
            zenbu_yaku[49] = 13

        # 50:小四喜
        if self.map[27] + self.map[28] + self.map[29] + self.map[30] == 11:
            zenbu_yaku[50] = 13

        # 役満以外
        # 8:断么九
        if sum([self.map[i] for i in Agari.YAOCHU1]) == 0:
            zenbu_yaku[8] = 1

        # 34:混一色, 35:清一色
        for i in [0, 9, 18]:
            if sum(self.map[i:i + 9]) == 14:
                zenbu_yaku[35] = 5 + self.menzen
            elif sum(self.map[i:i + 9] + self.map[27:34]) == 14:
                zenbu_yaku[34] = 2 + self.menzen

        # 10-13:自風, 14-17:場風, 18-20:役牌
        if self.map[27 + (self.ba[3] - self.ba[0]) % 4] == 3:
            zenbu_yaku[10 + (self.ba[3] - self.ba[0]) % 4] = 1

        if self.map[27 + (self.ba[0] // 4)] == 3:
            zenbu_yaku[14 + (self.ba[0] // 4)] = 1

        if self.map[31] == 3:
            zenbu_yaku[18] = 1
        if self.map[32] == 3:
            zenbu_yaku[19] = 1
        if self.map[33] == 3:
            zenbu_yaku[20] = 1

        # 30:小三元
        if self.map[31] + self.map[32] + self.map[33] == 8:
            zenbu_yaku[30] = 2

        # 31:混老頭
        if sum([self.map[i] for i in Agari.YAOCHU1]) == 14:
            zenbu_yaku[31] = 2

        return zenbu_yaku

    # 部分役
    def get_bubun_yaku(self, agari):
        bubun_yaku = [0 for i in range(0, 55)]

        # 役満
        # 40:四暗刻
        if agari.count(-4) + agari.count(-16) == 4:
            bubun_yaku[40] = 13

        # 51:四槓子
        if agari.count(-8) + agari.count(-16) == 4:
            bubun_yaku[51] = 13

        # 役満以外
        # 9:一盃口, 32:二盃口
        if self.menzen:
            if agari[0:3] == agari[3:6] and agari[6:9] == agari[9:12]:
                bubun_yaku[32] = 3
            elif agari[0:3] == agari[3:6] or agari[3:6] == agari[6:9] or agari[6:9] == agari[9:12]:
                bubun_yaku[9] = 1

        # 23:混全帯幺九, 33:純全帯幺九
        if self.zenbu_yaku[31] != 2:
            for i in range(0, 3 * 4, 3):
                if agari[i + 2] <= -2 and agari[i] in Agari.YAOCHU1:
                    continue
                if agari[i + 2] >= -1 and agari[i] in Agari.YAOCHU2:
                    continue
                break
            else:
                if agari[12] in Agari.YAOCHU1:
                    if sum(self.map[27:]) == 0:
                        bubun_yaku[33] = 2 + self.menzen
                    else:
                        bubun_yaku[23] = 1 + self.menzen

        # 24:一気通貫
        for i in [0, 9, 18]:
            for j in [0, 3, 6]:
                for k in range(0, 3 * 4, 3):
                    if agari[k + 2] >= -1 and agari[k] == i + j:
                        break
                else:
                    break
            else:
                bubun_yaku[24] = 1 + self.menzen
                break

        # 25:三色同順
        for i in combinations([agari[i] for i in range(0, 3 * 4, 3) if agari[i + 2] >= -1], 3):
            if i[0] % 9 == i[1] % 9 == i[2] % 9 and i[0] != i[1] and i[0] != i[2] and i[1] != i[2]:
                bubun_yaku[25] = 1 + self.menzen
                break

        # 26:三色同刻
        for i in combinations([agari[i] for i in range(0, 3 * 4, 3) if agari[i + 2] <= -2 and agari[i] <= 26], 3):
            if i[0] % 9 == i[1] % 9 == i[2] % 9 and i[0] != i[1] and i[0] != i[2] and i[1] != i[2]:
                bubun_yaku[25] = 2
                break

        # 27:三槓子
        if agari.count(-8) + agari.count(-16) == 3:
            bubun_yaku[27] = 2

        # 28:対々和
        if sum([1 for i in range(2, 3 * 4, 3) if agari[i] <= -2]) == 4:
            bubun_yaku[28] = 2
            print(agari)

        # 29:三暗刻
        if sum([1 for i in range(2, 3 * 4, 3) if agari[i] == -4 or agari[i] == -16]) == 3:
            bubun_yaku[29] = 2

        return bubun_yaku
