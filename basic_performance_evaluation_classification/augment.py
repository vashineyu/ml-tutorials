import imgaug as ia
import imgaug.augmenters as iaa


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

augment = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        rotate=(-15, 15), # rotate by -15 to +15 degrees
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    iaa.OneOf(
        [iaa.Add((-30, 30), per_channel=True),
         iaa.Multiply((0.9,1.1), per_channel=True)])
])