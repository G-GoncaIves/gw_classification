import torch
from torchvision.transforms.functional import autocontrast, equalize, solarize, affine, rotate
from torchvision.transforms import ColorJitter
from torchaudio.transforms import FrequencyMasking, TimeMasking
import numpy as np

class RandAug():

	def __init__(self, n, m, desired_transforms=None):

		implemented_transforms = {
		"Identity" : self._identity, 
		"Rotate" : self._rotate,  
		"ShearX" : self._shearx, 
		"ShearY" : self._sheary, 
		"TranslateX" : self._translatex, 
		"TranslateY" : self._translatey,
		"SpecAug" : self._specaug
		}

		if desired_transforms is not None:
			assert all([name in implemented_transforms.keys() for name in desired_transforms])

		else:
			desired_transforms = list(implemented_transforms.keys())

		self.transforms = [implemented_transforms[name] for name in desired_transforms]
		self.n = int(n)
		self.m = int(m)

	def _randaugment(self):

		random_transforms = np.random.choice(self.transforms, self.n)
		return random_transforms


	def __call__(self, sample):

		image, label = sample
		self.shape = image.shape
		image = torch.reshape(image, (-1, *self.shape))
		image = image.double()

		for augment in self._randaugment():

			image, label = augment(image, label)

		image = torch.reshape(image, self.shape)

		return image, sample


	def _identity(self, image, label):

		return image, label

	def _rotate(self, image, label):

		image = rotate(image, angle=self.m*60)
		return image, label

	def _shearx(self, image, label):

		image = affine(image, shear=(self.m, 0), angle=0, translate=(0,0), scale=1)
		return image, label

	def _sheary(self, image, label):

		image = affine(image, shear=(0, self.m), angle=0, translate=(0,0), scale=1)
		return image, label

	def _translatex(self, image, label):

		image = affine(image, translate=(self.m*2, 0), angle=0, shear=(0,0), scale=1)
		return image, label

	def _translatey(self, image, label):

		image = affine(image, translate=(0, -self.m*2), angle=0, shear=(0,0), scale=1)
		return image, label

	def _specaug(self, image, label):

		time_bins, freq_bins = self.shape
		freq_mask = FrequencyMasking(freq_bins*self.m/60)
		time_mask = TimeMasking(time_bins*self.m/60)

		image = freq_mask(image)
		image = time_mask(image)
		
		return image, label