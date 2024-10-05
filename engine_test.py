from utils.engine import Engine
from utils.animegan import AnimeGAN

if __name__ == '__main__':
    for model in ['Hayao_64', 'Hayao-60', 'Paprika_54', 'Shinkai_53']:
        animegan = AnimeGAN(f"models/{model}.onnx")
        engine = Engine(image_path="data/husky.jpg", show=True, output_extension=str(model), custom_objects=[animegan])
        engine.run()