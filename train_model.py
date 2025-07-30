import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Danh sách các file và nhãn tương ứng
path_data = "workplace_dataset/"

files_labels = {
    "ngoi_lam_viec.csv": 0,
    "ngoi_nga_lung.csv": 1,
    "nam_ngu.csv": 2,
    "gac_chan.csv": 3,
    "dung_day.csv": 4,
    "di_lai.csv": 5
}

X = []
y = []
no_of_timesteps = 10  # Số bước thời gian (sequence length)

# Đọc và xử lý dữ liệu từ mỗi file
for file, label in files_labels.items():
    df = pd.read_csv(path_data + file)
    dataset = df.iloc[:, :].values  # Lấy toàn bộ cột làm input
    n_sample = len(dataset)
    
    # Chia dữ liệu thành chuỗi thời gian
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(label)

X, y = np.array(X), np.array(y)
print(f"Dữ liệu đầu vào: {X.shape}, Nhãn: {y.shape}")

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Xây dựng mô hình cải tiến LSTM
model = Sequential([
    Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(units=128)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(6, activation="softmax")  # 6 lớp cho 6 nhãn
])

# Compile mô hình
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# EarlyStopping và Checkpoint
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_lstm_model.keras", save_best_only=True)

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác: {accuracy * 100:.2f}%")

# Lưu mô hình
model.save("lstm_pose_model.keras")

print("Huấn luyện xong và đã lưu model!")