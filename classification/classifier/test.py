from comet_ml import ExistingExperiment
from dotenv import dotenv_values
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

test_img_path = '/home/natty/invasive-animal-detection/data/processed/crops/864839041745064/56742447_e7b8aae08ccfb3f7fcccc75daaa87060.jpg___crop_64acde667a4fa20008180b32.jpg'

# init comet experiement
env_vars = {key: val for key, val in dotenv_values('.env').items()}
experiment = ExistingExperiment(
  api_key=env_vars['COMET_API_KEY'],
  experiment_key='4a9c536259544f1d8f797c37d0c0bc65'
)

# experiment.log_metric("test_metric", 10)

transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
img_as_pil = Image.open(test_img_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
img_as_tensor = transform(img_as_pil)

tensor_to_pil_transform = ToPILImage()
back_to_pil = tensor_to_pil_transform(img_as_tensor)

experiment.log_image(back_to_pil, name='test-back_to_pil')
