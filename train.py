import torch
import torch.optim as optim
import argparse
from TSRNet import TSRNet
from dataloader import load_data
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def main(data_path, sr, batch_size, learning_rate, epochs):
    # 데이터 로딩
    train_loader = load_data(data_path, sr, batch_size)

    # 모델 초기화
    model = TSRNet()

    # 손실함수 및 옵티마이저 설정
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 모델 학습
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data
            optimizer.zero_grad()

            # 모델 예측 및 손실 계산
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')


if __name__ == '__main__':
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='BCG Anomaly Detection Training')

    # 하이퍼파라미터를 위한 인자 추가
    parser.add_argument('--data_path', type=str, default='path_to_bcg_data.npy', help='Path to the BCG data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--sr', type=int, default=100, help='Sampling rate for BCG signals')

    # 인자 파싱
    args = parser.parse_args()

    # main 함수 호출
    main(args.data_path, args.batch_size, args.learning_rate, args.epochs, args.sr)