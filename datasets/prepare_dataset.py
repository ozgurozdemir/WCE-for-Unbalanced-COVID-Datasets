from dataset_factory import *
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(description="Prepare and save the datasets...")

parser.add_argument('-path',    default="./data",  type=str,  help="path of datasets folder")
parser.add_argument('-dataset', default="Covid5K", type=str,
  help="type of dataset (Covid5K/ChestXray/CheXpert). for ChestXray dataset image folder and metadata.csv is required.")

parser.add_argument('-pos_aug', default=False,     type=bool, help="apply augmentation on positive samples")
parser.add_argument('-neg_aug', default=False,     type=bool, help="apply augmentation on negative samples")
parser.add_argument('-test',    default=False,     type=bool, help="prepare the set of Covid5K")

args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(args.path):
        raise Exception("Given dataset path is either wrong or not exists...")

    # Loading dataset
    if args.test:
        dataset = Covid5K(args.path, name="Covid5K", train=False, augmentation=False, from_numpy=False)
        test_labels = [1., 0.] * len(dataset.healthy_images) + [0., 1.] * len(dataset.covid_images)

        positive_images = np.zeros((len(dataset.covid_images), 256, 256, 3))
        negative_images = np.zeros((len(dataset.healthy_images), 256, 256, 3))

        for i, image in enumerate(dataset.covid_images):
            image = dataset.normalize(np.array(image))
            positive_images[i] = tf.image.resize(image, (256, 256), antialias=True)

        for i, image in enumerate(dataset.healthy_images):
            image = dataset.normalize(np.array(image))
            negative_images[i] = tf.image.resize(image, (256, 256), antialias=True)

        test_inputs = np.concatenate([negative_images, positive_images])
        np.savez_compressed(f"{args.path}/Covid5K/test/test_inputs.npz", test_inputs)
        np.savez_compressed(f"{args.path}/Covid5K/test/test_labels.npz", test_labels)


    elif args.dataset is "Covid5K":
        dataset = Covid5K(args.path, name="Covid5K", train=True, augmentation=args.pos_aug, from_numpy=False)
    elif args.dataset is "ChestXray":
        dataset = ChestXray(args.path, name="ChestXray", from_numpy=False)
    elif args.dataset is "CheXpert":
        dataset = ChestXray(args.path, name="ChestXray", from_numpy=False)
    else:
        raise Exception("Dataset type is not given correctly. It should be either 'Covid5K', 'ChestXray' or 'CheXpert'")


    # Saving dataset
    if args.pos_aug and not os.path.exists(f"{dataset.path}/{dataset.name}/{dataset.positive_dir}_aug.npz"):
        dataset.save_dataset(as_numpy=True, augmentation=args.pos_aug, neg_augment=args.neg_aug)
    elif not os.path.exists(f"{dataset.path}/{dataset.name}/{dataset.positive_dir}.npz"):
        dataset.save_dataset(as_numpy=True, augmentation=args.pos_aug, neg_augment=args.neg_aug)

    if args.neg_aug and not os.path.exists(f"{dataset.path}/{dataset.name}/{dataset.negative_dir}_aug.npz"):
        dataset.save_dataset(as_numpy=True, augmentation=args.pos_aug, neg_augment=args.neg_aug)
    elif not os.path.exists(f"{dataset.path}/{dataset.name}/{dataset.negative_dir}.npz"):
        dataset.save_dataset(as_numpy=True, augmentation=args.pos_aug, neg_augment=args.neg_aug)
