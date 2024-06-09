# 顔検出モデルのバイアス分析

## リポジトリ概要
このリポジトリでは、論文"[Are Face Detection Models Biased?](https://arxiv.org/abs/2211.03588)"に基づき、複数の顔検出モデル（MTCNN, BlazeFace, DSFD, RetinaFace）を使用し、CelebAデータセットを使用しバイアス分析を行います。

## 目的と目標
顔検出モデルにおけるバイアスの有無を確認し、その結果をもとに改善策を提案することを目的としています。

## 使用モデル
- [MTCNN](https://github.com/ipazc/mtcnn)
- [BlazeFace](https://github.com/hollance/BlazeFace-PyTorch)
- [DSFD](https://github.com/Tencent/FaceDetection-DSFD)
- [RetinaFace](https://github.com/serengil/retinaface)

## 使用データセット
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## セットアップと実行手順
**1. リポジトリのクローン**
```bash
git clone -b main https://github.com/aattyy-102030/Thesis_validation.git
cd Thesis_validation
```

**2.  以下のコマンドを使用して必要なライブラリをインストールしてください。**
```bash
pip install -r requirements.txt
```

**3. データセットをダウンロードし、data/フォルダに配置します。**
<br>
<br>

## 実行方法
**1. データの前処理を行います:**
```bash
python src/data_loader.py
```
<br>

**2. 各モデルで顔検出を行います:**
```bash
python main.py
```
<br>

**3. 結果を評価します:**
```bash
python src/evaluation.py
```
- [ ] 出力例：(画像貼る)

## 結果の確認
結果はresults/フォルダに保存されます。<br>
各モデルの出力・バイアスの評価結果を確認してください。

## 今後の展望
- [ ] コードの検証、結果の考察
