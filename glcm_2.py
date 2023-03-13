import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Verilerin yüklenmesi ve etiketlerin tanımlanması
img_size = (768, 512)  # Resim boyutu
num_classes = 8  # Sınıf sayısı
data = []
labels = []
for i in range(num_classes):
    folder_name = 'images/e' + str(i)
    for j in range(1, 101):
        img_name = str(j) + '.tif'
        img_path = folder_name + '/' + img_name
        img = io.imread(img_path, as_gray=True)
        img = img[:img_size[0], :img_size[1]]  # Resim boyutunu ayarlama
        data.append(img)
        labels.append(i)

# GLCM özelliklerinin çıkarılması
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['contrast', 'homogeneity', 'energy', 'correlation']
glcm_features = []
for img in data:
    glcm = greycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    glcm_features.append(feats)
glcm_features = np.array(glcm_features)

# Veri setini eğitim ve test kümelerine ayırma
split_ratio = 0.8
split_idx = int(split_ratio * len(labels))
train_X = glcm_features[:split_idx]
train_y = labels[:split_idx]
test_X = glcm_features[split_idx:]
test_y = labels[split_idx:]

# Sınıflandırıcı modelinin eğitimi ve test edilmesi
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(train_X, train_y)
pred_y = rfc.predict(test_X)
accuracy = accuracy_score(test_y, pred_y)
print('Accuracy:', accuracy)