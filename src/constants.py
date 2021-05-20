
version = "v.0.1.15-cuda"


def get_type_by_name(name):
    if name in cell_lines:
        return 'cell line'
    elif name in drugs:
        return 'drug'
    else:
        raise ValueError("Unknown cell line or drug: {}".format(name))


cell_lines = ['ACHN', 'HT29', 'M14',  # batch 1
              'IGROV1', 'MDAMB231', 'SF539',   # batch 2
              'HS578T', 'SKMEL2', 'SW620',  # batch 3
              'EKVX', 'OVCAR4', 'UACC257',  # batch 4
              'BT549', 'LOXIMVI', 'MALME3M',  # batch 5
              'A498', 'COLO205', 'HOP62',  # batch 6
              'HCT15', 'OVCAR5', 'T47D']  # batch 7

drugs = ['Chlormethine', 'Clofarabine', 'Panzem-2-ME2', 'Pemetrexed', 'Asparaginase',
         'Irinotecan', 'Gemcitabine', '17-AAG', 'Docetaxel', 'Erlotinib',
         'UK5099', 'Fluorouracil', 'Everolimus', 'MEDICA 16', 'BPTES',
         'Oligomycin A', 'Trametinib', 'Oxaliplatin', 'Rapamycin', 'Etomoxir',
         'Lenvatinib', 'Oxfenicine', 'Mercaptopurine', 'Metformin', 'Omacetaxine',
         'Cladribine', 'Paclitaxel', 'Methotrexate', 'PBS', 'Topotecan',
         'YC-1', 'Decitabine']