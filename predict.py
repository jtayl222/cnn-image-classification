import argparse
import json
import torch
from PIL import Image
from torchvision.transforms import transforms

from load_model import load_model
from utils import process_image


def process_image(image, test_transforms):
    try:
        with Image.open(image) as im:
            # im.show()
            im = test_transforms(im)
            # print(im.shape)
            # numpy_array = im.numpy().transpose(1,2,0)
            # print(numpy_array.shape)
            return im
    except FileNotFoundError:
        print(f"Error: File not found: {image}")
        return None
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        return None

def predictor(image_path, model, topk=5, device='cuda'):
    try:
        print("image_path: ", image_path)
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        pt_image = process_image(image_path, test_transforms)
        if pt_image is None:
            return None, None
        images = pt_image[None, :, :, :]
        model.to(device)
        images = images.to(device)
        model.eval()
        output = model.forward(images)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_class_labels = [model.classes[c] for c in top_class.data.tolist()[0]]
        return top_p.data.tolist()[0], top_class_labels
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        return None, None
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def main(
        path_to_image="flowers/valid/95/image_07512.jpg",
        checkpoint="checkpoint.pth",
        top_k=5,
        category_names="cat_to_name.json",
        device=None):

    model = load_model("vgg16", checkpoint, None, train_loader=None)
    top_p, top_class = predictor(path_to_image, model, top_k, device)
    if top_p is None:
        print("Prediction failed.")
        return
    print(top_p)
    #print(top_class)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    labels = x = list(map(cat_to_name.get, top_class))
    print(labels)

###
### Ref: https://docs.python.org/3/howto/argparse.html
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image")
    parser.add_argument("checkpoint")
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu")
    args = parser.parse_args()

    if args.gpu:
        device = "cuda:0"
    else:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Calling main()")
    main(
        path_to_image=args.path_to_image,
        checkpoint=args.checkpoint,
        top_k=args.top_k,
        category_names=args.category_names,
        device=device,
    )