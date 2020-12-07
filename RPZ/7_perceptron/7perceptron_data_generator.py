import numpy as np
import matplotlib.pyplot as plt


def click(max_number_of_clicks, timeout):
    print("\nPlease click.")
    coordinates = plt.ginput(max_number_of_clicks, timeout=timeout)
    print("Stop")
    print("Coordinates: ")
    print(coordinates)
    return coordinates


def tuple2np(coordinates):
    dt = np.dtype('float,float')
    arr = np.array(coordinates, dtype=dt)
    # X_coord = np.vstack((arr['f0'], arr['f1']))
    X_coord = arr['f0']
    y_coord = np.zeros(np.size(X_coord))
    return X_coord, y_coord


if __name__ == "__main__":

    # choose background picture
    filename = 'sc.png'
    img = plt.imread(filename)

    # define coordinate boundaries
    xmin, xmax, ymin, ymax = [0, 10, 0, 5]

    # data for class 1
    plt.imshow(img, extent=[xmin, xmax, ymin, ymax])
    coordinates_1 = click(max_number_of_clicks=150, timeout=200)
    X1, y1 = tuple2np(coordinates_1)

    # data for class 2
    plt.imshow(img, extent=[xmin, xmax, ymin, ymax])
    coordinates_2 = click(max_number_of_clicks=150, timeout=200)
    X2, y2 = tuple2np(coordinates_2)

    # concatenate all data and save
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2 + 1))
    np.savez('my_data.npz', X=X, y=y)
