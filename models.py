from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float
from sqlalchemy.orm import relationship

from database import Base

class User(Base):
    __tablename__ = 'user'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    loginId = Column(String(30), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    name = Column(String(30), nullable=False)
    
    dogs = relationship('Dog', back_populates='user')
    tokens = relationship('RefreshToken', back_populates='user')

class Dog(Base):
    __tablename__ = 'dog'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userId = Column(Integer, ForeignKey('user.id'))
    dogName = Column(String(255), nullable=False)
    breed = Column(String(255), nullable=False)
    breedCategory = Column(Integer, nullable=False)
    dogAge = Column(Integer, nullable=False)
    sex = Column(String(20), nullable=False)
    weight = Column(Float, nullable=False)
    
    user = relationship('User', back_populates='dogs')
    pictures = relationship('Picture', back_populates='dog')
    senseDatas = relationship('SenseData', back_populates='dog')
    targetExercise = relationship('TargetExercise', back_populates='dog')
    exerciseLogs = relationship('ExerciseLog', back_populates='dog')
    sequences = relationship('Sequence', back_populates='dog')

class Picture(Base):
    __tablename__ = 'picture'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dogId = Column(Integer, ForeignKey('dog.id'))
    fileName = Column(String(255), nullable=False)
    contentType = Column(String(128), nullable=False)
    photoPath = Column(String(255), nullable=False)

    dog = relationship('Dog', back_populates='pictures')

class SenseData(Base):
    __tablename__ = 'senseData'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dogId = Column(Integer, ForeignKey('dog.id'))
    measureTime = Column(DateTime(timezone=True), nullable=False)
    ax = Column(Integer, nullable=False)
    ay = Column(Integer, nullable=False)
    az = Column(Integer, nullable=False)
    bcg = Column(Integer, nullable=False)
    gx = Column(Integer, nullable=False)
    gy = Column(Integer, nullable=False)
    gz = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)

    dog = relationship('Dog', back_populates='senseDatas')

class RefreshToken(Base):
    __tablename__ = 'refreshToken'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userId = Column(Integer, ForeignKey('user.id'))
    token = Column(String(500), nullable=False)
    createdAt = Column(DateTime(timezone=True), nullable=False)
    expiresAt = Column(DateTime(timezone=True), nullable=False)

    user = relationship('User', back_populates='tokens')

class Sequence(Base):
    __tablename__ = 'sequence'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dogId = Column(Integer, ForeignKey('dog.id'))
    startTime = Column(DateTime(timezone=True), nullable=False)
    endTime = Column(DateTime(timezone=True), nullable=False)
    intentsity = Column(Integer, nullable=False)
    excercise = Column(Float, nullable=False)
    heartAnomoly = Column(Integer, nullable=False)
    heartRate = Column(Integer, nullable=False)
    respirationRate = Column(Integer, nullable=False)
    
    dog = relationship('Dog', back_populates='sequences')
    bcgdatas = relationship('Bcgdata', back_populates='sequence')

class Bcgdata(Base):
    __tablename__ = 'bcgData'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sequenceId = Column(Integer, ForeignKey('sequence.id')) 
    measureTime = Column(DateTime(timezone=True), nullable=False)
    heart = Column(Float, nullable=False)
    respiration = Column(Float, nullable=False)

    
    sequence = relationship('Sequence', back_populates='bcgdatas')

class TargetExercise(Base):
    __tablename__ = 'targetExercise'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dogId = Column(Integer, ForeignKey('dog.id'))
    target = Column(Float, nullable=False)
    today = Column(Float, nullable=False)
    
    dog = relationship('Dog', back_populates='targetExercise')

class ExerciseLog(Base):
    __tablename__ = 'exerciseLog'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dogId = Column(Integer, ForeignKey('dog.id'))
    date = Column(DateTime(timezone=True), nullable=False)
    exercise = Column(Float, nullable=False)
    
    dog = relationship('Dog', back_populates='exerciseLogs')