import numpy as np
import pyvista as pv


def render_volume(data, label, pred, dimension=3):
    # dumps a dummy .mp4 at tmp to later upload it to w&b
    path_to_file = "/tmp/dummy.mp4"
    if dimension == 3:
        data = data.squeeze(0).squeeze(0).permute(2, 0, 1).numpy().astype(np.float32)
        label = label.squeeze(0).squeeze(0).permute(2, 0, 1).numpy().astype(np.float32)
        pred = pred.squeeze(0).squeeze(0).permute(2, 0, 1).numpy().astype(np.float32)
    else:
        data = data.squeeze(1).permute(2, 0, 1).numpy().astype(np.float32)
        label = label.squeeze(1).permute(2, 0, 1).numpy().astype(np.float32)
        pred = pred.squeeze(1).permute(2, 0, 1).numpy().astype(np.float32)

    for i in range(label.shape[2]):
        temp = label[:, :, i]
        temp2 = pred[:, :, i]
        label[:, :, i] = np.where(temp == 1.0, i, 0.0)
        pred[:, :, i] = np.where(temp2 == 1.0, i, 0.0)

    opacity = [0, 0.1, 0.3, 0.6, 1]
    p = pv.Plotter(off_screen=True)
    p.camera_position = [
        (-290.57860746006133, -372.55905040248126, 177.16263044387514),
        (132.68710699994833, 47.08861333109958, 98.96608072755437),
        (0.09095595121999676, 0.09301484361724968, 0.991501514776717),
    ]

    p.add_volume(data, cmap="bone", opacity=opacity, show_scalar_bar=False)
    p.add_volume(
        label,
        cmap="Reds",
        shade=True,
        show_scalar_bar=False,
        pickable=True,
        # ambient=0.5,
    )
    p.add_volume(
        pred,
        cmap="Greens",
        shade=False,
        show_scalar_bar=False,
        pickable=True,  # , ambient=0.5
    )
    path = p.generate_orbital_path(n_points=40, viewup=(0, 0, 1))
    p.show(auto_close=False)
    p.open_movie(path_to_file)
    p.orbit_on_path(path, step=0.01, write_frames=True)
    p.close()
    return path_to_file
