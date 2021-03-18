
import os, shutil, numpy
from PIL import Image
from tqdm import tqdm
from src.constants import user


class DataManager:

    data_path = '/Users/{}/ETH/projects/morpho-learner/data/HT29_CL1_P1/'.format(user)
    remote_data_path = '/Volumes/biol_imsb_sauer_1/users/Andrei/pheno-ml/cell_line_images/batch_1/HT29_CL1_P1/'

    def __init__(self, path=None):
        if path is not None:
            self.data_path = path

    def copy_random_images(self, N):
        """ This method copies N images from Sauer 1 to a local machine. """

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        for path in tqdm(os.listdir(self.remote_data_path)[:N]):
            if not path.startswith('.') and  path.endswith('.jpg'):
                shutil.copyfile(self.remote_data_path + path, self.data_path + path)

    @staticmethod
    def cut_images_and_save(in_folder, out_folder, n_piecies):
        """ This method cuts images in N square pieces and saves to a new location. """

        for name in tqdm(os.listdir(in_folder)):
            if not name.startswith('.') and name.endswith('.jpg'):

                image = Image.open(in_folder + name)

                a = numpy.array(image)  # get an array to crop the image

                one_row = numpy.sqrt(n_piecies).astype('int')
                for i in range(one_row):
                    for j in range(one_row):

                        cropped_array = a[(i*a.shape[0] // one_row):((i+1)*a.shape[0] // one_row),
                                        (j*a.shape[0] // one_row):((j+1)*a.shape[0] // one_row)]

                        cropped_image = Image.fromarray(cropped_array)  # get back the image object
                        cropped_image.save(out_folder + name.replace('.jpg', '_{}.jpg'.format(i*one_row+j+1)), "JPEG")


if __name__ == "__main__":

    dm = DataManager()
    dm.cut_images_and_save('/Users/dmitrav/ETH/projects/morpho-learner/data/HT29_CL1_P1_controls/',
                           '/Users/dmitrav/ETH/projects/morpho-learner/data/cut_controls/', 16)