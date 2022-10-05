import numpy as np

from python_utils.mathu import normang

sinPolyCoef3 = -1.666665710e-1                                          # Double: -1.666665709650470145824129400050267289858e-1
sinPolyCoef5 =  8.333017292e-3                                          # Double:  8.333017291562218127986291618761571373087e-3
sinPolyCoef7 = -1.980661520e-4                                          # Double: -1.980661520135080504411629636078917643846e-4
sinPolyCoef9 =  2.600054768e-6                                          # Double:  2.600054767890361277123254766503271638682e-6

TanCoef3 = 0.334961658
TanCoef5 = 0.118066350
TanCoef7 = 0.092151584

# From cleanflight: https://github.com/cleanflight/cleanflight/blob/master/src/main/common/maths.c
def sin_approx(x):
  while x > np.pi:
    x -= 2 * np.pi                                 # always wrap input angle to -PI..PI
  while x < -np.pi:
    x += 2 * np.pi

  if x > np.pi / 2:
    x = (np.pi / 2) - (x - np.pi / 2)   # We just pick -90..+90 Degree
  elif x < -np.pi / 2:
    x = -(np.pi / 2) - (np.pi / 2 + x)

  x2 = x * x
  return x + x * x2 * (sinPolyCoef3 + x2 * (sinPolyCoef5 + x2 * (sinPolyCoef7 + x2 * sinPolyCoef9)))
  #return x + x * x2 * (sinPolyCoef3)

def cos_approx(x):
  return sin_approx(x + np.pi / 2)

def tan_approx(x):
  x2 = x * x
  return x + x * x2 * (TanCoef3 + x2 * (TanCoef5 + x2 * TanCoef7))

def fastsin(arr):
  arr = normang(arr)

  # We just pick -90 to + 90 degrees.
  arr[arr >  np.pi / 2] =  np.pi / 2 - (arr[arr > np.pi / 2] - np.pi / 2)
  arr[arr < -np.pi / 2] = -np.pi / 2 - (np.pi / 2 + arr[arr < -np.pi / 2])

  x2 = arr * arr
  #return arr + arr * x2 * (sinPolyCoef3 + x2 * (sinPolyCoef5 + x2 * (sinPolyCoef7 + x2 * sinPolyCoef9)))
  #return arr + arr * x2 * (sinPolyCoef3 + x2 * (sinPolyCoef5 + x2 * (sinPolyCoef7)))
  return arr + arr * x2 * (sinPolyCoef3 + x2 * (sinPolyCoef5))
  #return arr + arr * x2 * (sinPolyCoef3)

def fastcos(arr):
  return fastsin(arr + np.pi / 2)
