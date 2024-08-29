from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from core.security import decode_access_token
from database import get_db
from routers.auth import verify_and_refresh_token
from schemas import SenseDataCreate, SequenceCreate, BcgdataCreate
from crud import create_sense_data, get_user_by_loginId, get_dog_by_user, get_dog_weight_by_user
from crud import create_sequence, create_bcgdata, update_today_exercise, get_bcgdata_by_sequence, get_sequences_asc_by_dog
from models import Sequence, Bcgdata
from aiModels.yeinOh import *
from aiModels.dongukKim import *
import pandas as pd
import numpy as np
import pickle

router = APIRouter()

global sensorDataBuffer
global bufferSize
sensorDataBuffer = []
bufferSize = 0

async def run_first_model(db, dog, websocket, input_datas, result):
    # 필요 데이터 나누기
    inputSequence = [list(data.values()) for data in input_datas]
    time = [data["time"] for data in input_datas]
    bcg = np.array([data["bcg"] for data in input_datas])
    
    # 모델 로직 - 동욱님 코드
    model_filename = 'aiModels/kmeans_model_newfinal.pkl'
    _, _, cluster, excerciseNum = process_data(inputSequence, model_filename, dog.weight)
    excerciseNum = float(excerciseNum/2) # 운동 값 절반 적용
    update_today_exercise(db, dog.id, excerciseNum)
    run_model = (cluster == 0)

    # 모델 함수 (수면 중일 때 이상치 탐지) - 예인님 코드
    anomalies_detected = False
    if run_model:
        bpm_h, bpm_r, combined_matrix_for_s, input = preprocess_data(time, bcg, run_model = True)
        model_path = "aiModels/TSRNet-33.pt"
        threshold = 0.0317
        anomalies_detected, _ = TRSNET(model_path, input, threshold)
    else: 
        # bpm_h = 심박수, bpm_r = 호흡수
        # combined_matrix_for_s = (time, filtered_hr, filtered_rp) = (시간, 심박, 호흡)
        bpm_h, bpm_r, combined_matrix_for_s, _ = preprocess_data(time, bcg)
        
    bpm_h = int(bpm_h)
    bpm_r = int(bpm_r)
    combined_matrix_for_s = combined_matrix_for_s[140:420]
    
    # 시퀀스 데이터 생성
    sqCreate = SequenceCreate(
        dogId = dog.id,
        startTime = combined_matrix_for_s[0][0],
        endTime = combined_matrix_for_s[-1][0],
        intentsity = cluster,
        excercise = excerciseNum,
        heartAnomoly = anomalies_detected,
        heartRate = bpm_h,
        respirationRate = bpm_r
    )
    sequenceData = create_sequence(db, sqCreate)
    
    bcgHeart = []
    # bcg 데이터 생성
    for data in combined_matrix_for_s:
        bcgObject = BcgdataCreate(
            sequenceId = sequenceData.id,
            measureTime = data[0],
            heart = float(data[1]),
            respiration = float(data[2])
        )
        bcgHeart.append({"time": bcgObject.measureTime.timestamp(), "heart": float(data[1])})
        create_bcgdata(db, bcgObject)

    # sequence 데이터와 bcg 데이터를 클라이언트로 전송
    await websocket.send_json({"heartRate": sequenceData.heartRate,
                               "respirationRate":sequenceData.respirationRate,
                               "heartAnomoly":False,
                               "senseData":bcgHeart,
                               "intentsity":sequenceData.intentsity,
                               "accessToken": result
                              })
    return

# 센서 데이터를 데이터베이스에 저장
async def upload_sense_data(db, dog_id, sense_data_list):
    for sense_data in sense_data_list:
        sensor_data_obj = SenseDataCreate(
                measureTime=sense_data["time"],
                ax=sense_data["ax"],
                ay=sense_data["ay"],
                az=sense_data["az"],
                bcg=sense_data["bcg"],
                gx=sense_data["gx"],
                gy=sense_data["gy"],
                gz=sense_data["gz"],
                temperature=sense_data["temperature"]
            )
        create_sense_data(db, sensor_data_obj, dog_id)

@router.websocket("/wsbt")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    global sensorDataBuffer
    global bufferSize
    await websocket.accept()
    
    try:
        # 첫 번째 메시지에서 액세스 토큰을 수신
        data = await websocket.receive_json()
        accessToken = data.get("accessToken")

        # 토큰 검증
        is_valid, result = verify_and_refresh_token(db, accessToken)
        if not is_valid:
            await websocket.send_json({"auth_success": False, "message": "Authentication fail", "accessToken": result})
            await websocket.close()
            return
        else:
            await websocket.send_json({"auth_success": True, "message": "Authentication success", "accessToken": result})

        payload = decode_access_token(result)
        
        # 사용자 정보에서 강아지 정보 조회
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            await websocket.send_json({"auth_success": False, "message": "Dog information does not exist", "accessToken": result})
            await websocket.close()
            return
        
        # 인증 후 수신된 데이터 처리
        while True:
            data = await websocket.receive_json()
            sensor_data_list = data.get("senserData")

            if not sensor_data_list:
                continue
            #await upload_sense_data(db, dog.id, sensor_data_list)
            sensorDataBuffer.extend(sensor_data_list)
            bufferSize += len(sensor_data_list)
            if bufferSize >= 560:
                modelInputDatas = sensorDataBuffer[:560]

                # 모델 실행
                await run_first_model(db, dog, websocket, modelInputDatas, result)

                # 데이터 버퍼 갱신
                sensorDataBuffer = sensorDataBuffer[280:]
                bufferSize -= 280

    except WebSocketDisconnect:
        print("Client disconnected")

# 시연용 웹소켓 : 자동으로 DB에 있는 데이터를 전송
@router.websocket("/test-wsbt")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()

    try:
        # 첫 번째 메시지에서 액세스 토큰을 수신
        data = await websocket.receive_json()
        accessToken = data.get("accessToken")

        # 토큰 검증
        is_valid, result = verify_and_refresh_token(db, accessToken)
        if not is_valid:
            await websocket.send_json({"auth_success": False, "message": "Authentication fail", "accessToken": result})
            await websocket.close()
            return
        else:
            await websocket.send_json({"auth_success": True, "message": "Authentication success", "accessToken": result})

        payload = decode_access_token(result)

        # 사용자 정보에서 강아지 정보 조회
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            await websocket.send_json({"auth_success": False, "message": "Dog information does not exist", "accessToken": result})
            await websocket.close()
            return

        dogSequences = get_sequences_asc_by_dog(db, dog.id)
        i = 0
        testLen = 0
        # 인증 후 수신된 데이터 처리
        while True:
            data = await websocket.receive_json()
            testInput = data.get("senserData")
            if not testInput:
                continue
            testLen += len(testInput)
            if testLen >= 560:
                sequenceData = dogSequences[i]
                bcgHeart = [{"time":bcg.measureTime.timestamp(), "heart":bcg.heart} for bcg in get_bcgdata_by_sequence(db, sequenceData.id)]
                await websocket.send_json({"heartRate": sequenceData.heartRate,
                                "respirationRate":sequenceData.respirationRate,
                                "heartAnomoly":False,
                                "senseData":bcgHeart,
                                "intentsity":sequenceData.intentsity,
                                "accessToken": result
                                })
                # 데이터 버퍼 갱신
                testLen -= 280
                i += 1

    except WebSocketDisconnect:
        print("Client disconnected")