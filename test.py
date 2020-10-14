from evaluation_metrics import *
from dataset_factory import *
from model_factory import *
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser(description="Test the pretrained network...")

parser.add_argument('-test_path',   default="./data/Covid5K/test",  type=str,    help="path of test set folder")

parser.add_argument('-model',   default="resnet50", type=str,   help="type of network, resnet50/acnn")
parser.add_argument('-acnn_block_size', default=16, type=str,   help="number of blocks in acnn")

parser.add_argument('-pretrain_path', default="./pretraining/chexpert_acnn",    type=str,
                                      help="path of pretrained network")
args = parser.parse_args()


TEST_INPUTS = np.load(args.test_path + "test_inputs.npz")['arr_0']
TEST_LABELS = np.load(args.test_path + "test_labels.npz")['arr_0']
TEST_LABELS = [np.argmax(i) for i in TEST_LABELS]


# Model initialization
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
pretrain_ckpt = None

if args.model == "resnet50":
    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(256, 256, 3),
        pooling=None, classes=2, classifier_activation='softmax')

    model = PreassembledCNN(resnet50, num_classes=2, optimizer=optimizer,
                            checkpoint_dir=args.pretrain_path, load_checkpoint=True)


if args.model == "acnn":
    model = ACNN(block_size=args.acnn_block_size, num_classes=2, optimizer=optimizer,
                 checkpoint_dir=args.pretrain_path, load_checkpoint=True)


# Test the network
predictions = []
for i, inp in tqdm(enumerate(TEST_INPUTS)):
    out = model(inp[tf.newaxis,...])
    predictions.append(np.argmax(out.numpy()))

log = experiment_report(TEST_LABELS, predictions)
print(log)

with open(f"{args.model}_{time.time()}.txt", "w") as file:
    file.write(str(args.__dict__) + "\n\n" + ("-"*50) + "\n\n" + log)
