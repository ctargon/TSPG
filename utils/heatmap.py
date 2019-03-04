import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcol
import matplotlib.cm as cm


def heatmap(arr):
	fig, ax = plt.subplots(1, 4)

	# Make a user-defined colormap.
	# cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
	cdict = {'red':  ((0.0, 0.0, 0.0),
				   (0.25, 0.0, 0.0),
				   (0.5, 0.8, 1.0),
				   (0.75, 1.0, 1.0),
				   (1.0, 0.4, 1.0)),

		 'green': ((0.0, 0.0, 0.0),
				   (0.25, 0.0, 0.0),
				   (0.5, 0.9, 0.9),
				   (0.75, 0.0, 0.0),
				   (1.0, 0.0, 0.0)),

		 'blue':  ((0.0, 0.0, 0.4),
				   (0.25, 1.0, 1.0),
				   (0.5, 1.0, 0.8),
				   (0.75, 0.0, 0.0),
				   (1.0, 0.0, 0.0))
	}
	blue_red = mcol.LinearSegmentedColormap('BlueRed1', cdict)
	plt.register_cmap(name="BlueRed", cmap=blue_red) 

	# cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)

	for i in range(4):
		im = ax[i].imshow(np.expand_dims(arr[i], -1), cmap="BlueRed")
		im.set_clim(-1,1)

		ax[i].set_xticks([])
		ax[i].set_xticklabels([])

		if i == 0:
			ax[i].set_yticks(np.arange(arr.shape[1]))
			ax[i].set_yticklabels(np.arange(arr.shape[1]) + 1)
		else:
			ax[i].set_yticks([])
			ax[i].set_yticklabels([])

	cbar = ax[i].figure.colorbar(im, ax=ax)
	cbar.ax.set_ylabel("Expression Level", rotation=-90, va="bottom")

	plt.show()



if __name__ == "__main__":
	# define some examples
	x = np.asarray([0., 0.0324, 0.0041, 0.1162, 0.0051, 0.0387, 0.0541, 0.0123, 0.0572, 0.0066,
					0.1536, 0.0064, 0.0688, 0.0131, 0.3311, 0.0659, 0.1737, 0.1332, 0.0001, 0.0002,
					0.2426, 0.1292, 0.084,  0.0129, 0.0769, 0.114, 0.0171, 0.0269, 0.026,  0.0371,
					0.2119, 0.0042, 0.0516, 0.0789, 0.2463, 0.1495])

	p = np.asarray([-0.6603, -0.492, 0.0248, -0.0227, -0.47, -0.7036, 0.0797, -0.8314, -0.9341,
				-0.5207, -0.7289, -0.8471, 0.0326, -0.9053, -0.7853, 0.0154, 0.0221, -0.5212,
				-0.9716, -0.9608, -0.1169, 0.133, 0.1286, 0.0184, -0.7416, 0.0122, 0.068,
				0.0493,  0.0907, -0.5981, -0.0411, 0.1083, -0.6401, -0.7304, -0.0069, -0.1115])

	x_hat = np.asarray([0.    , 0.    , 0.0289, 0.0935, 0.    , 0.    , 0.1338, 0.    ,
						0.    , 0.    , 0.    , 0.    , 0.1014, 0.    , 0.    , 0.0813,
						0.1958, 0.    , 0.    , 0.    , 0.1257, 0.2622, 0.2126, 0.0313,
						0.    , 0.1262, 0.1    , 0.0762, 0.1167, 0.    , 0.1708, 0.1125,
						0.    , 0.    , 0.2394, 0.038 ])

	mu_t = np.asarray([0.0001, 0.0243, 0.0346, 0.0933, 0.0001, 0.0141, 0.1446, 0.0124, 0.022, 0.016,
						0.0146, 0.0012, 0.0946, 0.0065, 0.0317, 0.1094, 0.2086, 0.0021, 0., 0.0001,
						0.1386, 0.2143, 0.2103, 0.033, 0.0134, 0.1262, 0.0846, 0.088, 0.0979, 0.0151,
						0.1629, 0.118,  0.036,  0.0318, 0.2358, 0.0638])

	results = np.vstack([x, p , x_hat, mu_t])

	heatmap(results)




