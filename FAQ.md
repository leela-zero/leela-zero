# Leela Zero常见问题解答 #
# Frequently Asked Questions about Leela Zero #

## 为什么网络不是每次都变强的 ##
## Why doesn't the network get stronger every time ##

从谷歌的论文中可以发现，AZ的网络强度也是有起伏的。而且现在只是在小规模测试阶段，发现问题也是很正常的。请保持耐心。

AZ's network isn't always be stronger either. Besides we're actually testing our approach right now. Just be patient.

## 为什么现在训练的是5/6 block网络，而AZ用的是20block ##
## Why the network size is only 6 blocks comparing to 20 blocks of AZ ##

在项目起步阶段，较小的网络可以在短时间内得到结果，也可以尽早发现/解决问题，

目前的主要目的是为了测试系统的可行性，这对今后的完整重现十分重要（为将来的大网络打好基础）。

This is effectively a testing run to see if the system works, and which things are important for doing a full run. I expected 10 to 100 people to run the client, not 600.

Even so, the 20 block version is 13 times more computationally expensive, and expected to make SLOWER progress early on. I think it's unwise to do such a run unless it's proven that the setup works, because you are going to be in for a very long haul.

## 为什么比较两个网络强弱时经常下十几盘就不下了 ##
## Why only dozens of games are played when comparing two networks ##

这里使用的是概率学意义上强弱，具体来说是SPRT在95%概率下任何一方有超过55%的胜率（ELO的35分），就认为有一方胜出了。谷歌的论文中是下满400盘的。唯一的区别是我们这里的Elo可能不是那么准确，网络的强弱还是可以确定的。

We use SPRT approach for best network choosing. A better network is chosen only SPRT finds it's 95% confidence to have a win rate of 55% (boils down to 35 Elo). Side effect is our Elo curve may not be that accurate, but the best network chosen is still correct.

## 自对弈时产生的棋谱为什么下得很糟 ##
## Why the game generated during self-play contains quite a few bad moves ##

生成自对弈棋谱时，使用的模拟步数(playout)只有1000，还加入了噪声，这是为了增加随机性，之后的训练才有进步的空间。如果用图形界面（如sabiki）加载Leela Zero，并设置好参数与之对弈，你会发现它其实表现得并不赖。

The playouts of self-play games is only 1000, and with noises added (For randomness of each move thus training has something to learn from). If you load Leela Zero with Sabaki, you'll probably find it is actually not that weak.
