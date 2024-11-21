#!/usr/bin/python
# coding: UTF-8

import pickle
import re
import time

import requests
from agari import *


def bit2dec(m):
    return sum([m[i] * (2**i) for i in range(len(m))])


def bit2huro(m):
    b = []
    for i in range(0, 16):
        b.append(m % 2)
        m = m // 2

    who = b[0] + b[1] * 2

    if who == 0:
        # 暗槓
        return [bit2dec(b[8:16]) // 4, -1, -16]

    else:
        # 順子
        if b[2] == 1:
            tmp = bit2dec(b[10:16])
            return [9 * (tmp // 21) + (tmp // 3) % 7, -(tmp % 3), -1]

        # 明刻
        elif b[3] == 1:
            return [bit2dec(b[9:16]) // 3, -who, -2]

        # 加槓
        elif b[4] == 1:
            return [bit2dec(b[9:16]) // 3, -1, -8]

        # 明槓
        else:
            return [bit2dec(b[8:16]) // 4, -1, -8]


year = 2016
date = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if (year % 4) == 0 and ((year % 100) != 0 or (year % 400) == 0):
    date[1] = 28

counter = 1
with open("last_run.pickle", "rb") as f:
    last = pickle.load(f)
# last = 0

for m in range(0, 12):
    for d in range(0, date[m]):
        with open(
            "./"
            + str(year)
            + "/scc"
            + str(year)
            + ("%02d" % (m + 1))
            + ("%02d" % (d + 1))
            + ".html",
            "r",
            encoding="UTF-8",
        ) as f:
            html = f.read()

        for para in re.findall(
            '\| 四鳳.喰赤. \| <a href="http://tenhou.net/0/\?log=(.*?)">', html
        ):
            if counter < last:
                counter += 1
                continue

            url = "http://e.mjv.jp//0/log/?" + para
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
            }
            req = requests.get(url, headers=headers, allow_redirects=True)
            xml = str(req.text)

            print("Load:", url, "#", counter)
            print("http://tenhou.net/0/?log=" + para)

            for tag in re.findall("<.*?>", xml):
                elem, attr = re.findall("<(.*?)[ /](.*?)/?>", tag)[0]
                attr = dict(re.findall('(\w*?)="(.*?)"', attr))

                if elem == "mjloggm":
                    if attr["ver"] != "2.3":
                        print("version error")
                        exit()

                elif elem == "AGARI":
                    if daburon == 0:
                        ba = [int(i) for i in attr["ba"].split(",")]
                    else:
                        ba = [0, 0]

                    who = int(attr["who"])
                    fromWho = int(attr["fromWho"])

                    info = [kyoku] + ba + [who, fromWho]

                    if "paoWho" in attr:
                        paoWho = int(attr["paoWho"])
                        if fromWho != paoWho:
                            info += [paoWho]

                    # 手牌
                    tehai = [0 for i in range(0, 34)]
                    for i in sorted([int(i) // 4 for i in attr["hai"].split(",")]):
                        tehai[i] += 1

                    # 副露
                    huro = []
                    if "m" in attr:
                        h = [bit2huro(int(i)) for i in attr["m"].split(",")]

                        for i in h:
                            huro += i

                    # 待ち
                    machi = int(attr["machi"]) // 4

                    # 点
                    ten = [int(i) for i in attr["ten"].split(",")]

                    # 収支
                    sc = [int(i) for i in attr["sc"].split(",")][1::2]

                    # 状況役
                    jokyo_yaku = [0 for i in range(0, 55)]
                    if "yaku" in attr:
                        yaku = [int(i) for i in attr["yaku"].split(",")]
                        for i in range(len(yaku) // 2):
                            if yaku[i * 2] in [0, 1, 2, 3, 4, 5, 6, 21, 52, 53, 54]:
                                jokyo_yaku[yaku[i * 2]] = yaku[i * 2 + 1]

                    elif "yakuman" in attr:
                        yakuman = [int(i) for i in attr["yakuman"].split(",")]
                        for i in yakuman:
                            if i in [37, 38]:
                                jokyo_yaku[i] = 13

                    # DEBUG
                    new_sc = Agari(tehai, huro, info, jokyo_yaku).get_ten(machi)

                    print(sc, new_sc)

                    if [i * 100 for i in sc] != new_sc:
                        han = 0
                        if "yaku" in attr:
                            yaku = [int(i) for i in attr["yaku"].split(",")]
                            for i in range(len(yaku) // 2):
                                han += yaku[i * 2 + 1]

                        elif "yakuman" in attr:
                            han += 13

                        print()
                        print("http://tenhou.net/3/?log=" + para)
                        print(req.url)
                        print(req.text)
                        print(info)
                        print(han, ten[0])
                        print("天鳳", [i * 100 for i in sc])
                        new_sc = Agari(tehai, huro, info, jokyo_yaku)
                        print("自作", new_sc.get_ten(machi))
                        for i in range(55):
                            if (
                                new_sc.jokyo_yaku[i]
                                + new_sc.bubun_yaku[i]
                                + new_sc.zenbu_yaku[i]
                                > 0
                            ):
                                print(
                                    Agari.YAKU[i],
                                    new_sc.jokyo_yaku[i]
                                    + new_sc.bubun_yaku[i]
                                    + new_sc.zenbu_yaku[i],
                                )
                        exit()

                    else:
                        daburon = 1

                elif elem == "INIT":
                    kyoku = [int(i) for i in attr["seed"].split(",")][0]
                    daburon = 0

            with open("last_run.pickle", "wb") as f:
                pickle.dump(counter, f)

            counter += 1
            time.sleep(10)

# すべて終了
with open("last_run.pickle", "wb") as f:
    pickle.dump(0, f)
