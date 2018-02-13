# Leela Zero常见问题解答 #
# Frequently Asked Questions about Leela Zero #

## 为什么网络不是每次都变强的 ##
## Why doesn't the network get stronger every time ##

从谷歌的论文中可以发现，AZ的网络强度也是有起伏的。而且现在只是在小规模测试阶段，发现问题也是很正常的。请保持耐心。

AZ also had this behavior, besides we're testing our approach right now. Please be patient.

## 为什么现在训练的是5/6 block网络，而AZ用的是20block ##
## Why the network size is only 6 blocks comparing to 20 blocks of AZ ##

在项目起步阶段，较小的网络可以在短时间内得到结果，也可以尽早发现/解决问题，

目前的主要目的是为了测试系统的可行性，这对今后的完整重现十分重要（为将来的大网络打好基础）。

This is effectively a testing run to see if the system works, and which things are important for doing a full run. I expected 10 to 100 people to run the client, not 600.

Even so, the 20 block version is 13 times more computationally expensive, and expected to make SLOWER progress early on. I think it's unwise to do such a run unless it's proven that the setup works, because you are going to be in for a very long haul.

## 为什么比较两个网络强弱时经常下十几盘就不下了 ##
## Why only dozens of games are played when comparing two networks ##

这里使用的是概率学意义上强弱，具体来说是SPRT在95%概率下任何一方有超过55%的胜率（ELO的35分），就认为有一方胜出了。谷歌的论文中是下满400盘的。唯一的区别是我们这里的Elo可能不是那么准确，网络的强弱还是可以确定的。

We use SPRT to decide if a newly trained network is better. A better network is only chosen if SPRT finds it's 95% confident that the new network has a 55% (boils down to 35 elo) win rate over the previous best network.

## 自对弈时产生的棋谱为什么下得很糟 ##
## Why the game generated during self-play contains quite a few bad moves ##

生成自对弈棋谱时，使用的MCTS模拟次数只有1000，还加入了噪声，这是为了增加随机性，之后的训练才有进步的空间。如果用图形界面（如sabiki）加载Leela Zero，并设置好参数与之对弈，你会发现它其实表现得并不赖。

The MCTS playouts of self-play games is only 1000, and with noises added (For randomness of each move thus training has something to learn from). If you load Leela Zero with Sabaki, you'll probably find it is actually not that weak.

## 自对弈为什么使用1000的模拟次数，而不是AZ的1600 ##
## For self-play, why use 1000 playouts instead of 1600 playouts as AZ ##

没人知道AZ的1600是怎么得到的。这里的1000是基于下面几点估计得到的：

1. 对于某一个选点，MCTS需要模拟几次才能得出概率结果。在开始阶段，每个选点的概率不会差太多，所以开始的360次模拟大概会覆盖整个棋盘。所以如果要让某些选点可以做几次模拟的话，大概需要2到3 x 360次的模拟。

2. 在computer-go上有人跑过7x7的实验，看到模拟次数从1000到2000的时候性能有提高。所以如果我们观察到瓶颈的时候，可能是可以考虑增加模拟次数。

3. 模拟次数太多会影响得到数据的速度。

Nobody knows. The Zero paper doesn't mention how they arrive at this number, and I know of no sound background to estimate the optimal. I chose it based on some observations:

a) For the MCTS to feed back search probabilities to the learning, it must be able to achieve a reasonable amount of look-ahead on at least a few variations. In the beginning, when the network is untrained, the move probabilities are not very extreme, and this means that the first 360~ simulations will be spent expanding every answer at the root. So if we want to expand at least a few moves, we probably need 2 to 3 x 360 playouts.

b) One person on computer-go, who ran a similar experiment on 7x7, reported that near the end of the learning, he observed increased performance from increasing the number from 1000 to 2000. So maybe this is worthwhile to try when the learning speed starts to decrease or flatten out. But it almost certainly isn't needed early on.

c) Obviously, the speed of acquiring data is linearly related to this setting.

So, the current number is a best guess based on these observations. To be sure what the best value is, one would have to rerun this experiment several times.

## 有些自对弈对局非常短 ##
## Very short self-play games ends with White win?! ##

自对弈的增加了随机性，一旦黑棋在开始阶段选择pass，由于贴目的关系，白棋有很大概率也选择pass获胜。短对局由此产生。

This is expected. Due to randomness of self-play games, once Black choose to pass at the beginning, there is a big chance for White to pass too (7.5 komi advantage for White). See issue #198 for defailed explanation.

## 对局结果错误 ##
## Wrong score? ##

Leela Zero使用Tromp-Taylor规则(详见<https://senseis.xmp.net/?TrompTaylorRules>)。虽然与中国规则一样贴7.5目，但为计算方便，并不去除死子。因此，结果与使用中国规则计算可能有所不同。不过，不去除死子并不影响模型的训练结果，因为双方会将死子自行提掉。

Leela Zero uses Tromp-Taylor rules (see https://senseis.xmp.net/?TrompTaylorRules). Although its komi is 7.5 as in Chinese rule, for simplicity, Tromp-Taylor rules do not remvoe dead stones. Thus, the result may be different from that calcuated using Chinese rule. However, keeping dead stones does not affect training results because both players are expected to capture dead stones themselves.
