import pandas as pd
import numpy as np
import pickle

# 운동 수치 계산 함수
# 여기서는 0: 수면, 1: 중간 강도, 2: 높은 강도, 3: 낮은 강도로 분류.
# 운동 수치 계산할때만 이렇게 하고, 출력할때는 숫자 순으로 출력되도록 하는 'cluster_mapping' 변수가 있음.
# 운동 수치 = 클러스터 강도 * 강아지 몸무게(kg) * 운동 시간(h)
def calculate_activity(cluster, duration, dog_weight):
    intensity_scores = {0: 1, 1: 6, 2: 10, 3: 3}
    return round(intensity_scores[cluster] * dog_weight * duration, 4)

# dog_weight(강아지 몸무게)는 DB에서 가져와야함.
def process_data(batch_data, model_path, dog_weight=20):
    # 모델 로드
    with open(model_path, 'rb') as file:
        kmeans = pickle.load(file)
    
    # 데이터 프레임 생성
    data = pd.DataFrame(batch_data, columns=['timestamp', 'ax', 'ay', 'az', 'bcg', 'gx', 'gy', 'gz','temperature'])

    # 특성 계산
    features = np.hstack([data[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].mean(), data[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].std()])
    features_reshaped = features.reshape(1, -1)
    
    # 클러스터링 수행
    cluster = kmeans.predict(features_reshaped)[0]

    # 운동 수치 계산
    # duration은 h 단위로 전환
    duration = 5.6 / 60
    activity_score = calculate_activity(cluster, duration, dog_weight)

    # 클러스터 매핑
    # 0: 수면, 1: 낮은 강도, 2: 중간 강도, 3: 높은 강도로 바꿔서 출력되도록.
    cluster_mapping = {0: 0, 3: 1, 1: 2, 2: 3}
    mapped_cluster = cluster_mapping[cluster]

    return data['timestamp'].iloc[0], data['timestamp'].iloc[-1], mapped_cluster, activity_score