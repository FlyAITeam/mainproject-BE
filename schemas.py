from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class PictureBase(BaseModel):
    fileName: str
    contentType: str
    photoPath: str

class PictureCreate(PictureBase):
    pass

class Picture(PictureBase):
    id: int
    dogId: int

    class Config:
        from_attributes = True

class SenseDataBase(BaseModel):
    measureTime: datetime
    ax: int
    ay: int
    az: int
    bcg: int
    gx: int
    gy: int
    gz: int
    temperature: float

class SenseDataCreate(SenseDataBase):
    pass

class SenseData(SenseDataBase):
    id: int
    dogId: int

    class Config:
        from_attributes = True

class DogBase(BaseModel):
    dogName: str
    breed: str
    breedCategory: int
    dogAge: int
    sex: str
    weight: float

class DogCreate(DogBase):
    pass

class Dog(DogBase):
    id: int
    userId: int
    pictures: List[Picture] = []
    senseDatas: List[SenseData] = []

    class Config:
        from_attributes = True

class RefreshTokenBase(BaseModel):
    token: str
    createdAt: datetime
    expiresAt: datetime

class RefreshTokenCreate(RefreshTokenBase):
    pass

class RefreshToken(RefreshTokenBase):
    id: int
    userId: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    loginId: str
    name: str

class UserCreate(UserBase):
    password: str

class UserCreateRequest(BaseModel):
    loginId: str
    password: str
    name: str

class User(UserBase):
    id: int
    dogs: List[Dog] = []
    tokens: List[RefreshToken] = []

    class Config:
        from_attributes = True

class SequenceBase(BaseModel):
    dogId: int
    startTime: datetime
    endTime: datetime
    intentsity: int
    excercise: float
    heartAnomoly: int
    heartRate: int
    respirationRate: int

class SequenceCreate(SequenceBase):
    pass

class Sequence(SequenceBase):
    id: int
    bcgdatas: List['Bcgdata'] = []

    class Config:
        from_attributes = True

class BcgdataBase(BaseModel):
    sequenceId: int
    measureTime: datetime
    heart: float
    respiration: float

class BcgdataCreate(BcgdataBase):
    pass

class Bcgdata(BcgdataBase):
    id: int

    class Config:
        from_attributes = True

class TargetExerciseBase(BaseModel):
    dogId: int
    target: int
    today: int

class TargetExerciseCreate(TargetExerciseBase):
    pass

class TargetExercise(TargetExerciseBase):
    id: int

    class Config:
        from_attributes = True

class ExerciseLogBase(BaseModel):
    dogId: int
    date: datetime
    exercise: float

class ExerciseLogCreate(ExerciseLogBase):
    pass

class ExerciseLog(ExerciseLogBase):
    id: int

    class Config:
        from_attributes = True