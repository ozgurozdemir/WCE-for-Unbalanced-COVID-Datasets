from evaluation_metrics import *
from datasets.dataset_factory import *
from models.model_factory import *
from models.resnet import ResNet
from models.acnn import ACNN
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser(description="Prepare and save the datasets...")

parser.add_argument('-path',        default="./data",               type=str,    help="path of datasets folder")
parser.add_argument('-test_path',   default="./data/Covid5K/test",  type=str,    help="path of test set folder")

parser.add_argument('-dataset', default="Merge",   type=str,
                                help="type of dataset (Covid5K/Covid5Kaug/ChestXray/CheXpert/Merge)")

parser.add_argument('-model',   default="resnet50", type=str,   help="type of network, resnet50/acnn")
parser.add_argument('-acnn_block_size', default=16, type=str,   help="number of blocks in acnn")

parser.add_argument('-pretrain',      default=None, type=str,
                                      help="pretraining weights, imagenet/chexpert (path is required)")
parser.add_argument('-pretrain_path', default="./pretraining/chexpert_acnn",    type=str,
                                      help="path of pretraining weights for chexpert")

parser.add_argument('-batch',    default=64,        type=int,    help="batch size for training")
parser.add_argument('-wce_b',    default=None,      type=float,  help="beta parameter for wce")
parser.add_argument('-epochs',   default=50,        type=int,    help="number of epochs the network will be trained")
parser.add_argument('-save',     default=False,     type=bool,   help="save the trained network")
parser.add_argument('-test',     default=True,      type=bool,   help="test the network once train completed")

args = parser.parse_args()


# Loading dataset
if args.dataset == "Covid5K":
    train_dataset = Covid5K(args.path, name="Covid5K", train=True, augmentation=False, from_numpy=True)
elif args.dataset == "Covid5Kaug":
    train_dataset = Covid5K(args.path, name="Covid5K", train=True, augmentation=True, from_numpy=True)
elif args.dataset == "ChestXray":
    train_dataset = ChestXray(args.path, name="ChestXray", from_numpy=True)
elif args.dataset == "ChestXpert":
    train_dataset = CheXpert(args.path, name="CheXpert", from_numpy=True)
elif args.dataset == "Merge":
    dataset1 = Covid5K(args.path, name="Covid5K", train=True, augmentation=False, from_numpy=True)
    dataset2 = ChestXray(args.path, name="ChestXray", from_numpy=True)

    train_dataset = MergeDataset(dataset1, dataset2, valid_split=0.2, name="Covid5K+ChestXray")
    del dataset1, dataset2 # for decreasing memory usage

TRAIN_DATASET, VALID_DATASET = train_dataset.prepare_dataset(batch=args.batch)


# Model initialization
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
pretrain_load = args.pretrain is not None
pretrain_ckpt = None

if args.model == "resnet50":
    if args.pretrain == None:
        resnet50 = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_tensor=None, input_shape=(256, 256, 3),
            pooling=None, classes=2, classifier_activation='softmax')

    elif args.pretrain == 'imagenet':
        resnet50 = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(256, 256, 3),
            pooling=None, classes=2, classifier_activation='softmax')

    elif args.pretrain == 'chexpert':
        pretrain_ckpt = args.pretrain_path
        resnet50 = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_tensor=None, input_shape=(256, 256, 3),
            pooling=None, classes=2, classifier_activation='softmax')

    model = PreassembledCNN(resnet50, num_classes=2, optimizer=optimizer, wce_beta=args.wce_b,
                            checkpoint_dir=pretrain_ckpt, load_checkpoint=pretrain_load)


if args.model == "acnn":
    if args.pretrain == 'chexpert':
        pretrain_ckpt = args.pretrain_path

    model = ACNN(block_size=args.acnn_block_size, num_classes=2, optimizer=optimizer, wce_beta=args.wce_b,
                 checkpoint_dir=pretrain_ckpt, load_checkpoint=pretrain_load)


# Train the network
best_model = model.fit(TRAIN_DATASET, VALID_DATASET, epochs=args.epochs, save=args.save)


# Test the network
if args.test:
    del TRAIN_DATASET, VALID_DATASET
    TEST_INPUTS = np.load(args.test_path + "test_inputs.npz")['arr_0']
    TEST_LABELS = np.load(args.test_path + "test_labels.npz")['arr_0']
    TEST_LABELS = [np.argmax(i) for i in TEST_LABELS]

    predictions = []
    for i, inp in tqdm(enumerate(TEST_INPUTS)):
        out = best_model(inp[tf.newaxis,...])
        predictions.append(np.argmax(out.numpy()))

    log = experiment_report(TEST_LABELS, predictions)
    print(log)

    with open(f"{args.model}_{time.time()}.txt", "w") as file:
        file.write(str(args.__dict__) + "\n\n" + ("-"*50) + "\n\n" + log)
