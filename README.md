# 顔検出モデルのバイアス分析

## ■リポジトリ概要
このリポジトリでは、論文"[Are Face Detection Models Biased?](https://arxiv.org/abs/2211.03588)"に基づき、複数の顔検出モデル（MTCNN, BlazeFace, DSFD, RetinaFace）を使用し、CelebAデータセットを使用しバイアス分析を行います。

## ■目的と目標
顔検出モデルにおけるバイアスの有無を確認し、その結果をもとに改善策を提案することを目的としています。

## ■使用モデル
- [MTCNN](https://github.com/ipazc/mtcnn)
- [BlazeFace](https://github.com/hollance/BlazeFace-PyTorch)
- [DSFD](https://github.com/Tencent/FaceDetection-DSFD)
- [RetinaFace](https://github.com/serengil/retinaface)

## ■使用データセット
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ([GoogleDrive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg))

## ■セットアップと実行手順
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

## ■実行方法
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
python src/model_evaluation.py
```
- [ ] 出力例：(画像貼る)

## ■実行フロー
**1. main.pyの実行開始**
<br>
プロジェクトのエントリーポイントです。このスクリプトが全てを調整し、他のスクリプトを適宜呼び出します。

**2. ディレクトリの設定と作成 (utils.py内のcreate_directories関数)**
<br>
predictions_dirとresults_dirというディレクトリが存在しない場合、これらを新たに作成します。

**3. 顔検出器の初期化 (face_detection.py内のFaceDetectionクラス)**
<br>
このクラスは、MTCNN, BlazeFace, DSFD, RetinaFaceモデルを初期化し、これらを使用して画像から顔を検出する機能を提供します。

**4. 画像の読み込みと顔検出 (utils.pyのload_image関数とFaceDetectionクラスのdetect_facesメソッド)**
<br>
data/images ディレクトリから画像ファイルを読み込み、顔検出器であるdetect_facesメソッドを使用して顔を検出します。

**5. 予測結果の保存 (utils.pyのsave_predictions関数)**
<br>
検出した顔の情報（バウンディングボックス）をファイルに保存します。これは後の評価ステップで使用されます。

**6. モデル評価 (model_evaluation.py内のModelEvaluationクラスのevaluate_modelメソッド)**
<br>
保存された予測結果とアノテーションデータを比較して、各モデルの精度、再現率、F1スコアを計算します。

**7. 結果の保存**
<br>
最終的な評価結果をresults/evaluation_results.csvに保存します。

**8. main.pyの実行終了**
<br>
全ての処理が完了した後、プログラムは終了します。

## ■結果の確認
結果はresults/フォルダに保存されます。<br>
各モデルの出力・バイアスの評価結果を確認してください。

## ■今後の展望
- [ ] コードの検証、結果の考察
