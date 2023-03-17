import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage import io, color

# Veri kümesini yükleme
class1 = io.imread_collection('GLCM/images/e0/*.tif')
class2 = io.imread_collection('GLCM/images/e1/*.tif')
class3 = io.imread_collection('GLCM/images/e2/*.tif')
class4 = io.imread_collection('GLCM/images/e3/*.tif')
class5 = io.imread_collection('GLCM/images/e4/*.tif')
class6 = io.imread_collection('GLCM/images/e5/*.tif')
class7 = io.imread_collection('GLCM/images/e6/*.tif')
class8 = io.imread_collection('GLCM/images/e7/*.tif')


# Özellik çıkarma fonksiyonu
def extract_features(image):
    # GLCM matrisinin hesaplanması
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    
    # Hesaplanacak özelliklerin belirlenmesi
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    asm = graycoprops(glcm, 'ASM').ravel()
    mean = np.mean(glcm).ravel()

    # LBP özelliğinin hesaplanması
    lbp = local_binary_pattern(image, P=8, R=1)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    lbp_hist = hist.ravel()

    # Tüm özelliklerin birleştirilmesi
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm, mean, lbp_hist])

    return features

def calculate_features(images):
    features = []
    for image in images:
        feature = extract_features(image)
        features.append(feature)
    return np.array(features)


# Özelliklerin hesaplanması
X_class1 = calculate_features(class1)
X_class2 = calculate_features(class2)
X_class3 = calculate_features(class3)
X_class4 = calculate_features(class4)
X_class5 = calculate_features(class5)
X_class6 = calculate_features(class6)
X_class7 = calculate_features(class7)
X_class8 = calculate_features(class8)

y_class1 = np.zeros(len(class1))
y_class2 = np.ones(len(class2))
y_class3 = np.full(len(class3), 2)
y_class4 = np.full(len(class4), 3)
y_class5 = np.full(len(class5), 4)
y_class6 = np.full(len(class6), 5)
y_class7 = np.full(len(class7), 6)
y_class8 = np.full(len(class8), 7)


#Veri kümesinin birleştirilmesi
X = np.vstack([X_class1, X_class2, X_class3, X_class4, X_class5, X_class6, X_class7, X_class8])
y = np.hstack([y_class1, y_class2, y_class3, y_class4, y_class5, y_class6, y_class7, y_class8])

#Eğitim ve test verilerinin bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest sınıflandırıcısı
clf = RandomForestClassifier(n_estimators=100, random_state=42)

#Sınıflandırma modelinin eğitilmesi
clf.fit(X_train, y_train)

#Eğitilen modelin test edilmesi
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)