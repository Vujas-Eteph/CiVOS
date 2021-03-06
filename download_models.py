import os
import gdown

os.makedirs('saves', exist_ok=True)

# Download the models
print('Downloading propagation model...')
gdown.download('https://drive.google.com/uc?id=19dfbVDndFkboGLHESi8DGtuxF1B21Nm8', output='saves/propagation_model.pth', quiet=False)

print('Downloading fusion model...')
gdown.download('https://drive.google.com/uc?id=1Lc1lI5-ix4WsCRdipACXgvS3G-o0lMoz', output='saves/fusion.pth', quiet=False)

print('Downloading interaction model...')
gdown.download('https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18s_itermask.pth', output='saves/coco_lvis_h18s_itermask.pth', quiet=False)

print('Done.')
