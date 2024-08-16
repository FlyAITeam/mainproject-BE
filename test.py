import torch
import argparse
from TSRNet import TSRNet
from dataloader import load_data

def main(data_path, sr, batch_size, model_path):
    # 데이터 로딩
    test_loader = load_data(data_path, sr, batch_size, shuffle=False)

    # 모델 초기화 및 로드
    model = TSRNet(enc_in=1)  # 입력 채널 수에 따라 설정
    model.load_state_dict(torch.load(model_path))

    # 모델 테스트
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            time_inputs = data['time_bcg'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            spec_inputs = data['spectrogram_bcg'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # 모델 예측
            outputs, _ = model(time_inputs, spec_inputs)
            
            # 여기서 출력을 기반으로 이상치 탐지 수행
            # 예를 들어, reconstruction error 계산 후 threshold와 비교
            reconstruction_error = torch.mean((outputs - time_inputs) ** 2, dim=1)
            # 이상치 감지 로직을 여기에 추가

            print(reconstruction_error)  # 또는 다른 방식으로 결과 출력

if __name__ == '__main__':
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='BCG Anomaly Detection Testing')

    # 하이퍼파라미터를 위한 인자 추가
    parser.add_argument('--data_path', type=str, default='path_to_bcg_test_data.npy', help='Path to the BCG test data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model file')
    parser.add_argument('--sr', type=int, default=100, help='Sampling rate for BCG signals')

    # 인자 파싱
    args = parser.parse_args()

    # main 함수 호출
    main(args.data_path, args.sr, args.batch_size, args.model_path)
