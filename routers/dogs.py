from fastapi import APIRouter, HTTPException, Depends, Request, status, Header, Body, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from database import get_db
from routers.auth import verify_and_refresh_token, decode_access_token
from crud import create_dog, get_user_by_loginId, create_picture, get_pictures_by_dog, get_dog_by_user, create_target_exercise, create_exercise_log, get_last_days_average_exercise
from crud import get_sequences_by_dog, get_bcgdata_by_sequence, get_user_by_loginId, get_dog_by_user, get_target_exercise, get_recent_sequences, update_target_exercise
from schemas import DogCreate, PictureCreate, TargetExerciseCreate, ExerciseLogCreate
from datetime import datetime, timedelta
import logging
from pydantic import ValidationError
import shutil
import os

router = APIRouter()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 강아지 정보 등록
@router.post("/dogs", status_code=status.HTTP_201_CREATED)
async def add_dog(request: Request, accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errorMessage": result}
        )

    try:
        user_data = await request.json()
        try:
            dog = DogCreate(**user_data)
        except ValidationError as e:
            error_messages = e.errors()
            errors = [item['loc'][0] for item in error_messages]
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": f"{errors} is a required field"}
            )

        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
        
        # 강아지 정보 생성
        db_dog = create_dog(db, dog, db_user.id)
        if not db_dog:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
        
        # 운동 목표량 정보 생성
        try:
            targetNum = 0
            if db_dog.breedCategory == 3: # 대형견
                targetNum = db_dog.weight * 2700
            elif db_dog.breedCategory == 2: # 중형견
                targetNum = db_dog.weight * 2790
            else: # 소형견
                targetNum = db_dog.weight * 2700
            target_exercise = TargetExerciseCreate(
                dogId=db_dog.id,
                target=targetNum,  # 기본 목표 운동량 설정 (예: 60분)
                today=0     # 오늘 운동량 초기화
            )
            create_target_exercise(db, target_exercise)
        except Exception as e:
            logger.error(f"Error creating target exercise: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Error creating target exercise"}
            )
        
        return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={"message": "Dog information added successfully"},
                headers={"accessToken": result}
        )
    except Exception as e:
        logger.error(f"Error adding dog: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 강아지 사진 업로드
@router.post("/dogs/photos", status_code=status.HTTP_201_CREATED)
async def upload_dog_photo(accessToken: str = Header(...), db: Session = Depends(get_db), image: UploadFile = File(...)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errorMessage": result}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
            
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )
        # 기존 사진이 있는지 확인
        existing_photo = get_pictures_by_dog(db, dog.id)
        if existing_photo:
            # 기존 파일 삭제
            if os.path.exists(existing_photo.photoPath):
                os.remove(existing_photo.photoPath)

            # 새 파일 경로 생성
            photo_path = f"photos/{dog.id}_profile"
            with open(photo_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # 기존 사진 정보 업데이트
            existing_photo.fileName = image.filename
            existing_photo.contentType = image.content_type
            existing_photo.photoPath = photo_path
            db.commit()

        else:
            # 사진 정보 데이터베이스에 저장 (새 사진)
            photo_path = f"photos/{dog.id}_profile"
            with open(photo_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            photo_data = PictureCreate(
                fileName=image.filename,
                contentType=image.content_type,
                photoPath=photo_path
            )
            logger.warning(f"{dog.id}")
            create_picture(db, photo_data, dog.id)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "Photo upload completed"},
            headers={"accessToken": result}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 강아지 정보 수정
@router.put("/dogs/me", status_code=status.HTTP_200_OK)
async def update_dog_info(
    accessToken: str = Header(...),
    db: Session = Depends(get_db),
    dogName: str = Body(...),
    breed: str = Body(...),
    breedCategory: int = Body(...),
    dogAge: int = Body(...),
    sex: str = Body(...),
    weight: float = Body(...)
):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errorMessage": result}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
        
        # 사용자의 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )

        # 강아지 정보 업데이트
        dog.dogName = dogName
        dog.breed = breed
        dog.breedCategory = breedCategory
        dog.dogAge = dogAge
        dog.sex = sex
        dog.weight = weight

        db.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Dog information change successfully"},
            headers={"accessToken": result}
        )

    except ValidationError as e:
        error_messages = e.errors()
        errors = [item['loc'][0] for item in error_messages]
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"errorMessage": f"{errors} is a required field"}
        )
    except Exception as e:
        logger.error(f"Error updating dog: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 강아지 정보 조회
@router.get("/dogs/me", status_code=status.HTTP_200_OK)
async def get_dog_info(accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errorMessage": "Invalid token"}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
        
        # 사용자의 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )

        # 성공 시 강아지 정보 반환
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "dogName": dog.dogName,
                "breed": dog.breed,
                "breedCategory": dog.breedCategory,
                "dogAge": dog.dogAge,
                "sex": dog.sex,
                "weight": dog.weight
            },
            headers={"accessToken": result}
        )

    except Exception as e:
        logger.error(f"Error fetching dog information: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 강아지 사진 가져오기
@router.get("/dogs/photos", response_class=FileResponse)
async def get_dog_photo(accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errorMessage": result}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )
            
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )
        
        # 사진 정보 가져오기
        existing_photo = get_pictures_by_dog(db, dog.id)
        if not existing_photo or not os.path.exists(existing_photo.photoPath):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"errorMessage": "Photo not found"}
            )

        # 사진 파일 반환
        headers = {"accessToken": result}
        return FileResponse(path=existing_photo.photoPath, media_type=existing_photo.contentType, headers=headers)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 심박값 데이터 전송
@router.get("/hearts", status_code=status.HTTP_200_OK)
async def get_heart_data(accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errormessage": "Invalid token"}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )

        # 사용자에 해당하는 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )

        # 최신 시퀀스 정보 조회
        sequences = get_sequences_by_dog(db, dog.id)
        if not sequences:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Sequence information does not exist"}
            )
        
        latest_sequence = sequences[-1]  # 가장 최신 시퀀스

        # intensity 값에 따른 데이터 처리
        bcg_data = get_bcgdata_by_sequence(db, latest_sequence.id)
        bcg_data_list = [
            {"time": data.measureTime.timestamp(), "heart": data.heart} for data in bcg_data
        ]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "intensity": latest_sequence.intentsity,
                "bcgData": bcg_data_list
            },
            headers={"accessToken": result}
        )

    except Exception as e:
        logger.error(f"Error fetching heart data: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 운동 목표량 정보 전송
@router.get("/exercise", status_code=status.HTTP_200_OK)
async def get_exercise_data(accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errormessage": "Invalid token"}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )

        # 사용자에 해당하는 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )

        # TargetExercise 정보 조회
        target_exercise = get_target_exercise(db, dog.id)
        if not target_exercise:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Target exercise information does not exist"}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "target": target_exercise.target,
                "today": target_exercise.today
            },
            headers={"accessToken": result}
        )

    except Exception as e:
        logger.error(f"Error fetching exercise data: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

# 1시간 시퀀스 데이터 전송
@router.get("/sequences", status_code=status.HTTP_200_OK)
async def get_sequences(accessToken: str = Header(...), db: Session = Depends(get_db)):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"errormessage": "Invalid token"}
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"errorMessage": "Server error"}
            )

        # 사용자에 해당하는 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Dog information does not exist"}
            )

        # 현재 시간으로부터 1시간 내의 시퀀스 정보 조회
        now = datetime.utcnow()
        sequences = get_recent_sequences(db, dog.id)
        
        if not sequences:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"errorMessage": "Sequence information does not exist"}
            )

        sequence_datas = [
            {
                "startTime": sequence.startTime.timestamp(),
                "endTime": sequence.endTime.timestamp(),
                "intensity": sequence.intentsity,
                "heartAnomoly": bool(sequence.heartAnomoly),
                "heartRate": sequence.heartRate,
                "respirationRate": sequence.respirationRate,
            }
            for sequence in sequences
        ]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"sequenceDatas": sequence_datas},
            headers={"accessToken": result}
        )

    except Exception as e:
        logger.error(f"Error fetching sequences: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"errorMessage": "Server error"}
        )

@router.get("/update-exercise", status_code=status.HTTP_200_OK)
async def update_exercise_and_target(
    accessToken: str = Header(...),
    db: Session = Depends(get_db)
):
    # 토큰 검증
    is_valid, result = verify_and_refresh_token(db, accessToken)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    try:
        # Access Token에서 로그인 ID 추출
        payload = decode_access_token(result)
        loginId = payload.get("sub")
        db_user = get_user_by_loginId(db, loginId)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User not found"
            )

        # 사용자에 해당하는 강아지 정보 조회
        dog = get_dog_by_user(db, db_user.id)
        if not dog:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dog information does not exist"
            )

        # 오늘의 운동량 가져오기
        target_exercise = get_target_exercise(db, dog.id)
        if not target_exercise:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target exercise not found"
            )
        returnTarget = target_exercise.target
        returnToday = target_exercise.today

        # 오늘의 운동량을 로그에 기록
        excerciseData = ExerciseLogCreate(
            dogId=dog.id,
            date=datetime.utcnow(),
            exercise=returnToday
        )
        create_exercise_log(db, excerciseData)

        # 오늘의 운동량 초기화
        target_exercise.today = 0
        db.commit()
        db.refresh(target_exercise)

        # 최근 5일간의 운동량 평균 계산
        average_exercise = get_last_days_average_exercise(db, dog.id, returnToday, returnTarget)
        # 목표 운동량 업데이트
        update_target_exercise(db, dog.id, average_exercise)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"target": returnTarget, "today": returnToday},
            headers={"accessToken": result}
        )

    except Exception as e:
        logger.error(f"Error updating exercise: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error"
        )