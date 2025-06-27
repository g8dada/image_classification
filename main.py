from src.train import train_model
import argparse
from src.config import CONFIG



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'inference'], required=True)
    parser.add_argument('--image', type=str, help='Path to image for inference', default='')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()    
    
    elif args.mode == 'inference':
        from src.inference import load_model, preprocess_image, predict
        
        # model = load_model('checkpoints/best_model.pt')
        model = load_model('checkpoints/model_epoch_20.pt')
        
        image_tensor = preprocess_image(args.image)
        class_names = CONFIG['classes']
        result = predict(model, image_tensor, class_names)
        print(f"Predicted class: {result}")

