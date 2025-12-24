import torch
import torchvision
import models
from PIL import Image
import numpy as np
import argparse
import time
from thop import profile


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def measure_model_complexity(model_hyper, sample_input):
    """Measure FLOPs and parameters using thop"""
    print("\n" + "="*60)
    print("Model Complexity Measurement")
    print("="*60)

    # Measure HyperNet
    flops_hyper, params_hyper = profile(model_hyper, inputs=(sample_input,), verbose=False)
    print(f"HyperNet FLOPs: {flops_hyper / 1e9:.4f} G")
    print(f"HyperNet Params: {params_hyper / 1e6:.4f} M")

    # Measure TargetNet
    with torch.no_grad():
        paras = model_hyper(sample_input)

    model_target = models.TargetNet(paras).cuda()
    target_input = paras['target_in_vec']

    flops_target, params_target = profile(model_target, inputs=(target_input,), verbose=False)
    print(f"TargetNet FLOPs: {flops_target / 1e9:.4f} G")
    print(f"TargetNet Params: {params_target / 1e6:.4f} M")

    total_flops = flops_hyper + flops_target
    total_params = params_hyper + params_target
    print(f"Total FLOPs: {total_flops / 1e9:.4f} G")
    print(f"Total Params: {total_params / 1e6:.4f} M")
    print("="*60)


def measure_throughput(model_hyper, sample_input, num_runs=100):
    """Measure inference throughput"""
    print("\n" + "="*60)
    print("Throughput Measurement")
    print("="*60)
    print(f"Running {num_runs} iterations...")

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            paras = model_hyper(sample_input)
            model_target = models.TargetNet(paras).cuda()
            pred = model_target(paras['target_in_vec'])

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = 1 / avg_time

    print(f"Average Inference Time: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("="*60 + "\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load HyperNet model
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    model_hyper.load_state_dict((torch.load(args.model_path, weights_only=False)))

    # Define image transforms
    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 384)),
                        torchvision.transforms.RandomCrop(size=224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

    # Load and prepare first image for complexity measurement
    img = pil_loader(args.image_path)
    img_tensor = transforms(img)
    img_tensor = img_tensor.to(device).unsqueeze(0)

    # Measure model complexity if requested
    if args.measure_complexity:
        measure_model_complexity(model_hyper, img_tensor)
        measure_throughput(model_hyper, img_tensor, num_runs=args.throughput_runs)

    # Random crop multiple patches and calculate mean quality score
    print("="*60)
    print("Quality Score Prediction")
    print("="*60)
    pred_scores = []
    for i in range(args.num_patches):
        img = pil_loader(args.image_path)
        img = transforms(img)
        img = img.to(device).unsqueeze(0)

        with torch.no_grad():
            paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = models.TargetNet(paras).to(device)

            # Quality prediction
            pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            pred_scores.append(float(pred.item()))

    score = np.mean(pred_scores)
    # quality score ranges from 0-100, a higher score indicates a better quality
    print(f'Predicted quality score: {score:.2f}')
    print("="*60)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HyperIQA: Image Quality Assessment Demo')
    parser.add_argument('--image_path', type=str, default='./data/D_01.jpg',
                        help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='./pretrained/koniq_pretrained.pkl',
                        help='Path to the pre-trained HyperIQA model weights')
    parser.add_argument('--num_patches', type=int, default=10,
                        help='Number of random patches to sample for quality prediction')
    parser.add_argument('--measure_complexity', action='store_true',
                        help='Measure model FLOPs, parameters, and throughput')
    parser.add_argument('--throughput_runs', type=int, default=100,
                        help='Number of iterations for throughput measurement')

    args = parser.parse_args()
    main(args)

