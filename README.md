# Mahjong
向聴数計算と点数計算と期待値計算

## 概要
agari.py - 点数計算を行う。<br>
agari_test.py - 天鳳の牌譜データを元にagari.pyの正しさを検証する。2016年度分、約18万半荘データ分で検証済み。<br>
exp.py - 期待値計算を行う。3向聴以上になると計算量が多くなり実用可能速度では求められないことが多い、向聴戻しを考慮しないなど改善の余地あり。<br>
make_shanten_table.py - 一色の面子手の向聴数を事前に計算してshanten_table.pickleとして出力する。<br>
shanten.py - shanten_table.pickleを利用して複数色の面子手の向聴数や七対子、国士無双の向聴数を求める。<br>
sim.py - 一人麻雀の打牌シミュレーションを行う。<br>
