import os
from torchvision import transforms, datasets



def get_loss(output, label):
    pass


def get_dataset(dataset_type, imgshape_2d):
    for folder in os.listdir(IMAGE_FOLDER):
        for file_name in os.listdir(os.path.join(IMAGE_FOLDER, folder)):
            pass


class TrainHelper:
    def __init__(self, img_shape, img_folder):
        self.img_shape = img_shape
        self.image_folder = img_folder
        self.transform = transforms.Compose(
            [transforms.Resize(imgshape_2d), transforms.ToTensor()])

    def get_dataset():
        train_data = datasets.I
            


def preprocessing(img, imgshape_2d):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_img = cv2.resize(
        gray_img, 
        (imgshape_2d[0], imgshape_2d[1]), 
        interpolation = cv2.INTER_AREA)
    reshaped_img = resized_img.reshape((resized_img.shape[0], resized_img.shape[1], 1))
    return reshaped_img



if __name__ == "__main__":
    print(os.listdir(os.getcwd()))