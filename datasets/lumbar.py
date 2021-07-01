import torchio as tio
from torchio.data import SubjectsDataset

def dataset_lumbar(
    data_list, transform, skip_list=[], path=""
):
    dataset = []
    for i in data_list:

        if i in skip_list:
            continue

        num = f"{i:02d}"
        subject = tio.Subject(
            data=tio.ScalarImage(path + num + "/vertebrae/FAT.dcm"),
            label=tio.LabelMap(path + num + "/vertebrae/L_all.mha"),
        )

        dataset.append(subject)
    return SubjectsDataset(dataset, transform=transform)


