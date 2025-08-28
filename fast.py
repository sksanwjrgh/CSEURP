# tf_check_fixed.py
import sys, time
import tensorflow as tf

print("Python :", sys.version.split()[0])
print("TF     :", tf.__version__)

# (선택) 독립 keras 패키지가 설치되어 있다면 버전 출력
try:
    import keras
    print("keras  :", keras.__version__)
except Exception:
    print("keras  : (standalone) not installed; using tf.keras")

# 사용 가능한 디바이스 나열
print("Physical devices:", tf.config.list_physical_devices())

# GPU 있으면 사용, 없으면 CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("WARN: set_memory_growth failed:", e)
    device_str = "/GPU:0"
else:
    device_str = "/CPU:0"
print("Using device:", device_str)

# 1) 큰 행렬곱(연산/성능 테스트)
N = 2048
dtype = tf.float32
with tf.device(device_str):
    a = tf.random.uniform((N, N), dtype=dtype)
    b = tf.random.uniform((N, N), dtype=dtype)
    t0 = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()  # 계산 강제
    t1 = time.time()
print(f"MatMul {N}x{N} took {t1 - t0:.3f} s on {device_str}")

# 2) tf.keras 간단 학습 테스트
with tf.device(device_str):
    x = tf.random.uniform((2048, 32), dtype=dtype)
    y = tf.random.uniform((2048, 1), dtype=dtype)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(32,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(x, y, epochs=1, batch_size=128, verbose=0)
print("Keras fit OK. Final loss:", float(hist.history["loss"][-1]))
