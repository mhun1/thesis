
import torchio as tio
from torchio.data import SubjectsDataset

def dataset_cervical(
    data_list, transform, skip_list=[], path=""
):
    dataset = []
    for i in data_list:
        if i in skip_list:
            continue

        num = f"{i:02d}"
        subject = tio.Subject(
            data=tio.ScalarImage(path + num + "/" + "data_" + num + ".nii.gz"),
            label=tio.LabelMap(path + num + "/" + "label_" + num + ".nii.gz"),
        )

        dataset.append(subject)
    return SubjectsDataset(dataset, transform=transform)
