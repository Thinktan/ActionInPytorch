import matplotlib.pyplot as plt
import torchvision.utils

from data import data_train, data_test

def plot_mnist():

    num_of_images = 60
    index=0

    for img, target in data_train:
        index += 1
        if index > num_of_images:
            break

        plt.subplot(6, 10, index)

        plt.axis('off')
        plt.imshow(img.numpy().squeeze(), cmap='gray_r')
    plt.show()

# images ,labels = next(iter(data_train_loader))
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img*std +mean
# print(labels)
# plt.imshow(img)
# plt.show()

# plot_mnist()

def test_one_case(index):
    img, target = data_test[index]
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
    print(target)
    plt.show()



