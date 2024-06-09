import os
import cv2

def load_image(image_path):
    """画像を読み込み、RGB形式に変換して返す"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_predictions(predictions, image_id, model_name, output_dir):
    """予測結果をテキストファイルに保存"""
    output_path = os.path.join(output_dir, f'{image_id}_{model_name}.txt')
    with open(output_path, 'w') as file:
        for box in predictions:
            line = ' '.join(map(str, box))
            file.write(f"{line}\n")

def create_directories(dir_list):
    """必要なディレクトリを作成"""
    for dir_path in dir_list:
        os.makedirs(dir_path, exist_ok=True)

def draw_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """画像にバウンディングボックスを描画"""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def save_image_with_boxes(image, boxes, output_path):
    """バウンディングボックス付きの画像を保存"""
    image_with_boxes = draw_boxes(image, boxes)
    cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

