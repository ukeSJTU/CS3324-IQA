import torch
import torchvision
import models
from PIL import Image
import numpy as np
import argparse


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def main(args):
    # Load HyperNet model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_hyper.load_state_dict((torch.load(args.model_path, weights_only=False)))

    # Define image transforms
    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 384)),
                        torchvision.transforms.RandomCrop(size=224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

    # Random crop multiple patches and calculate mean quality score
    pred_scores = []
    for i in range(args.num_patches):
        img = pil_loader(args.image_path)
        img = transforms(img)
        img = img.cuda().unsqueeze(0)
        paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

        # Building target network
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))

    score = np.mean(pred_scores)
    # quality score ranges from 0-100, a higher score indicates a better quality
    print('Predicted quality score: %.2f' % score)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HyperIQA: Image Quality Assessment Demo')
    parser.add_argument('--image_path', type=str, default='./data/D_01.jpg',
                        help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='./pretrained/koniq_pretrained.pkl',
                        help='Path to the pre-trained model weights')
    parser.add_argument('--num_patches', type=int, default=10,
                        help='Number of random patches to sample for quality prediction')

    args = parser.parse_args()
    main(args)

