import numpy as np
import torch

from deap_tools.operations import DimCompatible

# arr = raster_geometry.sphere((32,224,224), 1)
# print(arr.shape)
# plot in 3D
import matplotlib.pyplot as plt


# noise = np.random.normal(size=(224,224,224))
# print(noise.shape)
# print(noise)
# # fig = plt.figure()
# # ax = fig.add_subplot(1, 1, 1, projection='3d')
# #
# # verts, faces, normals, values = measure.marching_cubes(arr, 0.5)
# # ax.plot_trisurf(
# #     verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
# #     antialiased=False, linewidth=0.0)
# # plt.show()
#
# def create_bin_sphere(arr_size, center, r):
#     coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
#     distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2)
#     return 1*(distance <= r)
#
# ax = plt.axes(projection='3d')
#
# # Data for three-dimensional scattered points
# arr_size = (32,224,224)
# sphere_center = (15,15,15)
# r=10
# sphere = create_bin_sphere(arr_size,sphere_center, r)
#
# #Plot the result
# fig =plt.figure(figsize=(6,6))
# ax = fig.gca(projection='3d')
# ax.voxels(sphere, edgecolor='k')
# plt.show()
#
# def create_sphere(cx,cy,cz, r, resolution=360):
#     '''
#     create sphere with center (cx, cy, cz) and radius r
#     '''
#     phi = np.linspace(0, 2*np.pi, 2*resolution)
#     theta = np.linspace(0, np.pi, resolution)
#
#     theta, phi = np.meshgrid(theta, phi)
#
#     r_xy = r*np.sin(theta)
#     x = cx + np.cos(phi) * r_xy
#     y = cy + np.sin(phi) * r_xy
#     z = cz + r * np.cos(theta)
#
#     return np.stack([x,y,z])

# import pymesh
# mesh = pymesh.load_mesh("/home/mhun/git/common-3d-test-models/data/stanford_bunny.obj")
# mesh.vertices.shape
# import matplotlib.pyplot as plt
#
# X = np.arange(start=1, stop=51, step=1)
# y = np.arange(start=1, stop=51, step=1)
# kf = KFold(n_splits=4, shuffle=True, random_state=21)
# train_index, test_index = next(kf.split(X))
# print(train_index,test_index)
# dataset = get_dataset("Vertebrae3D", False, train_index, test_index)
# label = dataset["train"][0]["label"][tio.DATA].squeeze(0).float()


def pairwise(seq):
    items = iter(seq)
    last = next(items)
    for item in items:
        yield last, item
        last = item


def strictly_increasing(L):
    return all(x < y for x, y in pairwise(L))


def strictly_decreasing(L):
    return all(x > y for x, y in pairwise(L))


def non_increasing(L):
    return all(x >= y for x, y in pairwise(L))


def non_decreasing(L):
    return all(x <= y for x, y in pairwise(L))


def synthetic_data_test(label, func, steps=30, get_fig=True):
    noise = 2 * torch.randn((32, 224, 224))
    steps = 30

    score = []
    scores_norm = []
    for i in range(1, steps):
        alpha = 1 / i
        new_data = (1 - alpha) * label + alpha * noise
        new_data = torch.where(new_data > 0.3, 1.0, 0.0)
        # output = float(func(Prediction(new_data), Label(label)).val)

        output = float(func(DimCompatible(new_data), DimCompatible(label)).val)
        score.append(output)

    min_val = min(score)
    max_val = max(score)

    print("max: ", max_val)
    print("min: ", min_val)

    # (rawValue - min) / (max - min)

    if max_val - min_val != 0.0:
        for i in range(1, steps):
            alpha = 1 / i
            new_data = (1 - alpha) * label + alpha * noise
            new_data = torch.where(new_data > 0.3, 1.0, 0.0)
            # output = float(func(Prediction(new_data), Label(label)).val)
            output = (
                float(func(DimCompatible(new_data), DimCompatible(label)).val) - min_val
            ) / (max_val - min_val)
            scores_norm.append(output)
    else:
        out_dict = {
            "str_dec": False,
            "str_incr": False,
            "non_dec": False,
            "non_inc": False,
        }
        return score, out_dict, plt.figure(), min_val, max_val

    if get_fig:
        # x = np.linspace(1, 30, num=29)
        # fig = plt.figure()
        # plt.plot(x, score)
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show()

        x = np.linspace(1, 30, num=29)
        fig = plt.figure()
        plt.plot(x, scores_norm)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()

    out_dict = {
        "str_dec": strictly_decreasing(scores_norm),
        "str_incr": strictly_increasing(scores_norm),
        "non_dec": non_decreasing(scores_norm),
        "non_inc": non_increasing(scores_norm),
    }
    return score, out_dict, fig, min_val, max_val


def synthetic_load_data(func, get_fig=True):

    score = []
    scores_norm = []

    vals = [5, 1, 0, 4, 6, 8, 7, 10, 11, 12]
    for i in vals:

        name = "/home/mhun/store/syn_data/{}_{}.npy"
        # pred = torch.from_numpy(np.load(name.format(str(i),"pred")))
        sig = torch.from_numpy(np.load(name.format(str(i), "sig")))
        label = torch.from_numpy(np.load(name.format(str(i), "label")))

        output = float(func(DimCompatible(sig), DimCompatible(label)).val)
        # output = float(func(sig,label))
        score.append(output)

    # min_val = min(score)
    # max_val = max(score)
    #
    # print("max: ", max_val)
    # print("min: ", min_val)
    #
    # # (rawValue - min) / (max - min)
    #
    # if max_val - min_val != 0.0:
    #     for i in range(1, steps):
    #         alpha = 1 / i
    #         new_data = (1 - alpha) * label + alpha * noise
    #         new_data = torch.where(new_data > 0.3, 1.0, 0.0)
    #         #output = float(func(Prediction(new_data), Label(label)).val)
    #         output = (float(func(DimCompatible(new_data), DimCompatible(label)).val) - min_val) / (max_val-min_val)
    #         scores_norm.append(output)
    # else:
    #     out_dict = {"str_dec": False, "str_incr": False,
    #                 "non_dec": False, "non_inc": False}
    #     return score, out_dict, plt.figure(),min_val,max_val

    min_val = min(score)
    max_val = max(score)

    # print(score)
    if get_fig:
        x = np.linspace(0, 10, num=10)
        fig = plt.figure()
        plt.plot(x, score, label="loss")
        plt.plot(x, np.gradient(score), label="gradient")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()

        # x = np.linspace(1, 30, num=29)
        # fig = plt.figure()
        # plt.plot(x, scores_norm)
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show()

    out_dict = {
        "str_dec": strictly_decreasing(score),
        "str_incr": strictly_increasing(score),
        "non_dec": non_decreasing(score),
        "non_inc": non_increasing(score),
    }
    return score, out_dict, fig, min_val, max_val


# dice = Dice(apply_nonlin=False)
# _, convex, _, _, _ = synthetic_load_data(dice)
# print(convex)
# bce = BCE()
# _, convex, _, _, _ = synthetic_load_data(bce)
# print(convex)
# focal = Focal()
# _, convex, _, _, _ = synthetic_load_data(focal)
# print(convex)
