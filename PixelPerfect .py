import matplotlib.pyplot as plt
import numpy as np
# only used to read and save images
 
print("\n" + "=" * 50)
print("PROBLEM 2")
print("=" * 50)
 
# --- 2.1 ---
img = plt.imread("File 3.png")
print("2.1 shape:", img.shape)
print("2.1 dtype:", img.dtype)
print("2.1 smallest value:", img.min(), "| largest value:", img.max())
 
# fix the format
if img.shape[2] == 4:
    img = img[:, :, :3]  # keep only R, G, B and drop the transparency channel
if img.max() <= 1:
    # round first, otherwise values like 49.9999 would become 49 after uint8
    img = np.round(img * 255).astype(np.uint8)
 
print("2.1 shape after fix:", img.shape)
print("2.1 dtype after fix:", img.dtype)
print("2.1 range after fix:", img.min(), "to", img.max())
height, width, channels = img.shape
print("2.1 height:", height)
print("2.1 width:", width)
print("2.1 channels:", channels)
 
# --- 2.2 ---
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]
means = np.array([red.mean(), green.mean(), blue.mean()])
print("\n2.2 average Red:", round(means[0], 2))
print("2.2 average Green:", round(means[1], 2))
print("2.2 average Blue:", round(means[2], 2))
names = np.array(["Red", "Green", "Blue"])
print("2.2 strongest colour:", names[np.argmax(means)])
 
# --- 2.3 ---
# top-right quarter: rows from start to middle, columns from middle to end
crop = img[:height // 2, width // 2:, :]
print("\n2.3 crop shape:", crop.shape)
plt.imsave("crop.png", crop)
 
# --- 2.4 ---
# ::-1 reverses an axis. axis 1 is the width, so this flips left-to-right
mirror = img[:, ::-1, :]
plt.imsave("mirror.png", mirror)
print("\n2.4 mirror is different from the original:", bool(np.any(img != mirror)))
 
# --- 2.5 ---
weights = np.array([0.299, 0.587, 0.114])
# dot product between the last axis (3 channels) and the 3 weights
gray = np.dot(img, weights)
print("\n2.5 gray shape:", gray.shape)
# The dot product adds up the 3 colour values of every pixel into one number,
# so the channel dimension disappears and we lose one dimension.
# vmin/vmax make sure 0 = black and 255 = white when saving
plt.imsave("gray.png", gray, cmap="gray", vmin=0, vmax=255)
 
# --- 2.6 ---
# (a) the wrong version
bright_wrong = img + 60
plt.imsave("bright_wrong.png", bright_wrong)
# take one bright pixel (the position of the largest value in the image)
pos = np.unravel_index(np.argmax(img), img.shape)
print("\n2.6(a) one bright pixel before:", img[pos], "-> after image + 60:", bright_wrong[pos])
 
# (b) the correct version
# int16 can hold numbers bigger than 255, then clip forces them back to 0-255
bright_correct = np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8)
plt.imsave("bright_correct.png", bright_correct)
print("2.6(b) same pixel in the correct version:", bright_correct[pos])
 
# (c) uint8 can only store 0-255, so 255 + 60 = 315 wraps around to 59 and
# bright areas turn dark.
 
# --- 2.7 ---
# pixels brighter than 128 become 255 (white), everything else becomes 0 (black)
bw = np.where(gray > 128, 255, 0).astype(np.uint8)
white_pct = np.mean(bw == 255) * 100  # mean of True/False = fraction of True
print("\n2.7 percentage of white pixels: {:.2f}%".format(white_pct))
plt.imsave("blackwhite.png", bw, cmap="gray", vmin=0, vmax=255)
 
# --- 2.8 ---
# axis=1 puts the images next to each other (left and right)
side_by_side = np.concatenate((img, mirror), axis=1)
print("\n2.8 side by side shape:", side_by_side.shape)
# The width doubled because we joined two images of the same width
# along the width axis, so width = width + width.
plt.imsave("side_by_side.png", side_by_side)


