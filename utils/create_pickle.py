import pickle  
import numpy as np  
from PIL import Image  

# Open the PNG image
# img = Image.open('/mnt/f/Desktop/Josie_intepretable/cars/query_real.png')
img = Image.open('/mnt/f/Desktop/Josie_intepretable/birds/pied.png')
# img = Image.open('/mnt/f/Desktop/Josie_intepretable/birds/query_real.png')
# img = Image.open('/mnt/f/Desktop/Josie_intepretable/dogs/query_real.png')
# img = Image.open('/mnt/f/Desktop/Josie_intepretable/omniglot/query_real.png')
# Convert the image to RGB mode (removing the alpha channel)
img = img.convert('RGB')
# img = img.convert('L')
# Resize the image to 224x224
img = img.resize((224, 224))

# Convert the image to a numpy array with dtype as float64
img_array = np.array(img, dtype=np.float64)

# Create a dictionary
data_dict = {
    # 'rolls_and_buick': img_array
    'pied_kingfisher_and_savannah_sparrow': img_array
    # 'chow_and_shih_tzu': img_array
    # 'Japanese_(hiragana).character28_and_Sanskrit.character40': img_array
}

# Save the dictionary as a Pickle file
# with open('../data/rolls_and_buick.pkl', 'wb') as f:
with open('../data/pied.pkl', 'wb') as f:
# with open('../data/pied_kingfisher_and_savannah_sparrow.pkl', 'wb') as f:
# with open('../data/chow_and_shih_tzu.pkl', 'wb') as f:
# with open('../data/Japanese_(hiragana).character28_and_Sanskrit.character40.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("Pickle file has been successfully saved.")