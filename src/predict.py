import argparse
import fasttext
import cv2

def predict(model_path, image_path):
    clf = fasttext.load_model(model_path)
    image = cv2.imread(image_path)
    clf.predict(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--image")
    args = parser.parse_args()

    predict(args.model, args.image)
