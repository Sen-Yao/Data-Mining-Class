from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

print('开始读取')
# 步骤1：加载LFW数据集
lfw_people = fetch_lfw_people(data_home='../dataset', min_faces_per_person=50, resize=0.4)

print('读取完毕')

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]

# 标签为人脸对应的人名
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("总样本数: %d" % n_samples)
print("总特征数: %d" % n_features)
print("类别数: %d" % n_classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# 将原始特征转换为主成分特征
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# 步骤4：训练一个SVM分类器
svm = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svm)
model.fit(X_train, y_train)

# 步骤5：测试模型性能
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))
