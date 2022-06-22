from PIL import Image as I
import numpy as np
import pandas as pd

# get input image
ip_img = np.array(I.open('1data/hw3_part2_data/Penguins.jpg'))

print("ip_img", np.shape(ip_img))


ip_img = ip_img/255

# reshape to 2d array

points = np.reshape(ip_img, (ip_img.shape[0]*ip_img.shape[1], ip_img.shape[2]))


# uncomment if need to reproduce results same results
# np.random.seed(5)


# set value of k in k_means
k = 5

# randomly initialize centroids
centroid = np.random.rand(k, 3)


for i in range(15):
    # # calculate distances
    if(i != 0):
        distances = []
        for center in centroid:

            minus = center-points

            dist = np.linalg.norm(minus, axis=1)

            distances.append(dist)
        # return values of indices
        indices = np.argmin(distances, axis=0)
    else:
        indices = np.random.randint(k, size=(len(points)))

    unique_indices = np.unique(indices)

    # create combinations of indices and points
    points_index_df = pd.DataFrame(points, index=indices, columns=None)

    mean_df = pd.DataFrame(columns=[0, 1, 2])

    # calculate mean of the points in their respective clusters
    for j in range(0, k):
        # enter if that index has the value of points
        if (j in unique_indices):
            temp_df = points_index_df.loc[j, :]

            # if only one value is there for the index then mean is the point itself
            if(temp_df.shape == (3,)):
                temp_df = points_index_df.loc[j, :]
                temp_df = pd.DataFrame(
                    temp_df, columns=[j], index=None)

                mean_df = mean_df.append(temp_df.T)

            else:
                temp_df = temp_df.mean(axis=0)
                temp_df = pd.DataFrame(
                    temp_df, columns=[j], index=None)

                mean_df = mean_df.append(temp_df.T)

            del temp_df

    # update centroid

    for l in range(k):
        if (l in unique_indices):
            centroid[l] = mean_df.loc[l]

# assing resppective cluster values to form an output image
for m in range(len(centroid)):
    points_index_df.loc[m, 0] = centroid[m, 0]
    points_index_df.loc[m, 1] = centroid[m, 1]
    points_index_df.loc[m, 2] = centroid[m, 2]

out_img = points_index_df.to_numpy()

out_img = np.reshape(out_img, np.shape(ip_img))


# The output image will be saved to the current directory

im = I.fromarray(np.uint8(out_img*255)).convert('RGB')
im.save("Pengins_20.jpg")
print("Pengins_20.jpg", "created and saved")
