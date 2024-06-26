# 各ソースコードの概要
## data_loader.py
- DataLoaderクラス : 画像ディレクトリ、アノテーションファイル、出力ディレクトリを受け取る。
- process_imagesメソッド : アノテーションファイルを読み込み、各画像に対して前処理を行う。
- extract_facesメソッド : 各画像の顔の位置をアノテーションから取得する。
- save_processed_imageメソッド : 顔領域を切り出して保存する。

## face_detection.py
- FaceDetectionクラス: 複数の顔検出モデル（MTCNN、BlazeFace、DSFD、RetinaFace、ResNet）を初期化する。
- detect_facesメソッド : 指定されたモデル名に応じて適切な検出メソッドを呼び出す。
- その他メソッド : 各モデルに対して顔検出を行うメソッド（detect_faces_<model_name>）を定義する。

## model_evaluation.py
- calculate_iouメソッド : 2つのバウンディングボックス間のIoUを計算する。
- match_boxesメソッド : true_boxesとpred_boxesを比較して一致するかどうかを判定し、それに基づいて評価指標を計算する。
- evaluate_modelメソッド : 全ての画像についてIoUを計算し、マッチしたかどうかを基に精度、再現率、F1スコアを計算する。

## utils.py
- load_image: 指定したパスから画像を読み込み、RGB形式に変換して返す。
- save_predictions: 予測結果（バウンディングボックス）をテキストファイルに保存する。
- create_directories: 必要なディレクトリを作成する。
- draw_boxes: 指定したバウンディングボックスを画像に描画する。
- save_image_with_boxes: バウンディングボックス付きの画像を指定したパスに保存する。
