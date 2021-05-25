
import os, shutil, numpy, pandas
from PIL import Image
from tqdm import tqdm
from src.constants import drugs as all_drug_names


class DataManager:

    remote_data_path = '/Volumes/biol_imsb_sauer_1/users/Andrei/pheno-ml/cell_line_images/batch_1/HT29_CL1_P1/'

    def __init__(self, data_path=None, meta_path=None):
        if data_path is not None:
            self.data_path = data_path
        if meta_path is not None:
            self.meta_path = meta_path

    def copy_random_images(self, N, data_path=None):
        """ This method copies N images from Sauer 1 to a local machine. """

        if data_path is None:
            data_path = self.data_path

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for path in tqdm(os.listdir(self.remote_data_path)[:N]):
            if not path.startswith('.') and path.endswith('.jpg'):
                shutil.copyfile(self.remote_data_path + path, data_path + path)

    @staticmethod
    def cut_images_and_save(in_folder, out_folder, n_pieces):
        """ This method cuts images from in_folder in N square pieces and saves to a new location. """

        for name in tqdm(os.listdir(in_folder)):
            if not name.startswith('.') and name.endswith('.jpg'):

                image = Image.open(in_folder + name)

                a = numpy.array(image)  # get an array to crop the image

                one_row = numpy.sqrt(n_pieces).astype('int')
                for i in range(one_row):
                    for j in range(one_row):

                        cropped_array = a[(i*a.shape[0] // one_row):((i+1)*a.shape[0] // one_row),
                                        (j*a.shape[0] // one_row):((j+1)*a.shape[0] // one_row)]

                        cropped_image = Image.fromarray(cropped_array)  # get back the image object
                        cropped_image.save(out_folder + name.replace('.jpg', '_{}.jpg'.format(i*one_row+j+1)), "JPEG")

    def get_well_drug_mapping_for_cell_line(self, cell_line_folder, keep_max_conc_only=False):
        """ Retrieve a well-to-drug mapping for a cell line folder. """

        cell_line_meta = pandas.read_csv(self.meta_path + cell_line_folder + '.csv')
        wells = cell_line_meta['Well'].values
        drugs = cell_line_meta['Drug'].values
        concs = cell_line_meta['Final_conc_uM'].values
        del cell_line_meta

        if keep_max_conc_only:

            i = 0
            while i < len(drugs):

                if drugs[i] in [*all_drug_names, 'DMSO']:
                    # find max conc of the drug
                    drug_max_conc = max(concs[drugs == drugs[i]])
                    # remove a "column", if it's not related to max conc
                    if concs[i] != drug_max_conc:
                        wells = numpy.delete(wells, i)
                        drugs = numpy.delete(drugs, i)
                        concs = numpy.delete(concs, i)
                    else:
                        i += 1
                else:
                    i += 1

        mapping = {}
        for i in range(wells.shape[0]):
            mapping[wells[i]] = (drugs[i], concs[i])

        return mapping

    def process_images_of_single_well(self, path_to_images, well_files, n_pieces, out_folder, well, drug):

        for name in well_files:
            if not name.startswith('.') and name.endswith('.jpg'):

                image = Image.open(path_to_images + name)

                a = numpy.array(image)  # get an array to crop the image

                one_row = numpy.sqrt(n_pieces).astype('int')
                for i in range(one_row):
                    for j in range(one_row):

                        cropped_array = a[(i*a.shape[0] // one_row):((i+1)*a.shape[0] // one_row),
                                        (j*a.shape[0] // one_row):((j+1)*a.shape[0] // one_row)]

                        cropped_image = Image.fromarray(cropped_array)  # get back the image object

                        new_name = name.replace('.jpg', '_{}.jpg'.format(i*one_row+j+1))
                        new_name = new_name.replace('_{}_'.format(well), '_{}_{}_'.format(well, drug))

                        cropped_image.save(out_folder + new_name, "JPEG")

    def cut_images_from_all_batches_and_save(self, save_to_folder, n_pieces):
        """ This method cuts images from all batches in N square pieces and saves to a new location. """

        for b in tqdm(range(1,8)):
            batch_path = self.data_path + 'batch_{}/'.format(b)
            for cl_folder in tqdm(os.listdir(batch_path)):

                if not cl_folder.startswith('.'):
                    well_drug_map = self.get_well_drug_mapping_for_cell_line(cl_folder, keep_max_conc_only=True)
                    path_to_images = batch_path + cl_folder + '/'

                    for well in well_drug_map.keys():

                        drug = str(well_drug_map[well][0])

                        if drug == 'nan':
                            continue
                        elif drug == 'DMSO' or drug == 'PBS':
                            # controls
                            well_files = [file for file in os.listdir(path_to_images)
                                          if file.endswith('.jpg') and file.split('_')[3] == well]
                            well_files = sorted(well_files)[::3]  # get every third control image
                            out_path = save_to_folder + 'cut_controls/'

                        elif drug in all_drug_names:
                            # drugs
                            well_files = [file for file in os.listdir(path_to_images)
                                          if file.endswith('.jpg') and file.split('_')[3] == well]
                            well_files = sorted(well_files)[-10:]  # get last 10 drugs images
                            out_path = save_to_folder + 'cut/'
                        else:
                            raise ValueError("Unknown drug: {}".format(drug))

                        self.process_images_of_single_well(path_to_images, well_files, n_pieces, out_path, well, drug)

    def move_images_and_rename(self, save_to_folder, full_data=False):
        """ This method moves images to a single folder, renaming them to keep all meta info in a filename. """

        keep_max_conc_only = False
        control_step = None
        n_drug_images = None

        if not full_data:
            keep_max_conc_only = True
            control_step = None
            n_drug_images = None

        for b in tqdm(range(1,8)):
            batch_path = self.data_path + 'batch_{}\\'.format(b)
            for cl_folder in tqdm(os.listdir(batch_path)):

                if not cl_folder.startswith('.'):
                    well_drug_map = self.get_well_drug_mapping_for_cell_line(cl_folder, keep_max_conc_only=keep_max_conc_only)
                    path_to_images = batch_path + cl_folder + '\\'

                    for well in well_drug_map.keys():

                        drug = str(well_drug_map[well][0])
                        conc = str(well_drug_map[well][1])

                        if drug == 'nan':
                            continue
                        elif drug == 'DMSO' or drug == 'PBS':
                            # controls
                            well_files = [file for file in os.listdir(path_to_images)
                                          if file.endswith('.jpg') and file.split('_')[3] == well]

                            well_files = sorted(well_files)
                            if control_step is not None:
                                well_files = well_files[::control_step]  # get every nth control image

                        elif drug in all_drug_names:
                            # drugs
                            well_files = [file for file in os.listdir(path_to_images)
                                          if file.endswith('.jpg') and file.split('_')[3] == well]

                            well_files = sorted(well_files)
                            if n_drug_images is not None:
                                well_files = well_files[-n_drug_images:]  # get n last drugs images
                        else:
                            raise ValueError("Unknown drug: {}".format(drug))

                        # move images
                        for file in well_files:
                            new_filename = file.split('_')

                            new_filename[3] = '{}_{}_c={}'.format(new_filename[3], drug, conc)
                            new_filename = '_'.join(new_filename)

                            # renaming with drug and conc
                            shutil.copyfile(path_to_images + file,
                                            save_to_folder + new_filename)


if __name__ == "__main__":

    path_to_batches = 'D:\ETH\projects\morpho-learner\data\cropped\\'
    path_to_meta = 'D:\ETH\projects\morpho-learner\data\metadata\\'

    dm = DataManager(path_to_batches, path_to_meta)

    save_to_path = 'D:\ETH\projects\morpho-learner\data\cropped\\max_conc\\'
    dm.move_images_and_rename(save_to_path, full_data=False)

    save_to_path = 'D:\ETH\projects\morpho-learner\data\cropped\\full\\'
    dm.move_images_and_rename(save_to_path, full_data=True)
