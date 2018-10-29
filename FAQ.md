# Leela Zero常见问题解答 #
# Frequently Asked Questions about Leela Zero #

## 为什么网络不是每次都变强的 ##
## Why doesn't the network get stronger every time ##

从谷歌的论文中可以发现，AZ的网络强度也是有起伏的。而且现在只是在小规模测试阶段，发现问题也是很正常的。请保持耐心。

AZ also had this behavior, besides we're testing our approach right now. Please be patient.

## 为什么比较两个网络强弱时经常下十几盘就不下了 ##
## Why only dozens of games are played when comparing two networks ##

这里使用的是概率学意义上强弱，具体来说是SPRT在95%概率下任何一方有超过55%的胜率（ELO的35分），就认为有一方胜出了。谷歌的论文中是下满400盘的。唯一的区别是我们这里的Elo可能不是那么准确，网络的强弱还是可以确定的。

We use SPRT to decide if a newly trained network is better. A better network is only chosen if SPRT finds it's 95% confident that the new network has a 55% (boils down to 35 elo) win rate over the previous best network.

## 自对弈时产生的棋谱为什么下得很糟 ##
## Why the game generated during self-play contains quite a few bad moves ##

生成自对弈棋谱时，使用的MCTS模拟次数只有3200，还加入了噪声，这是为了增加随机性，之后的训练才有进步的空间。如果用图形界面（如sabiki）加载Leela Zero，并设置好参数与之对弈，你会发现它其实表现得并不赖。

The MCTS playouts of self-play games is only 3200, and with noise added (For randomness of each move thus training has something to learn from). If you load Leela Zero with Sabaki, you'll probably find it is actually not that weak.

## 有些自对弈对局非常短 ##
## Very short self-play games ends with White win?! ##

自对弈的增加了随机性，一旦黑棋在开始阶段选择pass，由于贴目的关系，白棋有很大概率也选择pass获胜。短对局由此产生。

This is expected. Due to randomness of self-play games, once Black choose to pass at the beginning, there is a big chance for White to pass too (7.5 komi advantage for White). See issue #198 for defailed explanation.

## 对局结果错误 ##
## Wrong score? ##

Leela Zero使用Tromp-Taylor规则(详见<https://senseis.xmp.net/?TrompTaylorRules>)。虽然与中国规则一样贴7.5目，但为计算方便，并不去除死子。因此，结果与使用中国规则计算可能有所不同。不过，不去除死子并不影响模型的训练结果，因为双方会将死子自行提掉。

Leela Zero uses Tromp-Taylor rules (see https://senseis.xmp.net/?TrompTaylorRules). Although its komi is 7.5 as in Chinese rule, for simplicity, Tromp-Taylor rules do not remove dead stones. Thus, the result may be different from that calcuated using Chinese rule. However, keeping dead stones does not affect training results because both players are expected to capture dead stones themselves.
