import os, pandas, time, torch, numpy, shutil, random, seaborn

if __name__ == "__main__":

    # save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'
    # comparison.plot_number_of_clusters('drugs', 300, save_to, filter_threshold=4)

    # train_control_path = 'D:\ETH\projects\morpho-learner\data\\train\\controls\\'
    # train_drug_path = 'D:\ETH\projects\morpho-learner\data\\train\\drugs\\'
    #
    # test_control_path = 'D:\ETH\projects\morpho-learner\data\\test\\controls\\'
    # test_drug_path = 'D:\ETH\projects\morpho-learner\data\\test\\drugs\\'
    #
    # controls = random.sample(os.listdir(train_control_path),
    #                                 int(0.1 * len(os.listdir(train_control_path))))
    #
    # drugs = random.sample(os.listdir(train_drug_path),
    #                                 int(0.1 * len(os.listdir(train_drug_path))))
    #
    # for control in controls:
    #     shutil.move(train_control_path + control, test_control_path + control)
    #
    # for drug in drugs:
    #     shutil.move(train_drug_path + drug, test_drug_path + drug)
    pass