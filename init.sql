-- .env파일에 맞게 수정 필요

-- 사용자 생성
-- CREATE USER DB_USER WITH PASSWORD DB_PASSWORD;
DO $$
BEGIN
    -- 사용자가 존재하는지 확인
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles 
        WHERE rolname = 'pelikan_test') THEN
        -- 사용자가 존재하지 않으면 생성
        CREATE USER pelikan_test WITH PASSWORD '1234';
        -- 데이터베이스를 생성 (데이터베이스 생성은 이 구문 밖에서 진행)
        -- 사용자에게 슈퍼유저 권한 부여
        ALTER ROLE pelikan_test SUPERUSER;
    END IF;
END$$;

-- 새 데이터베이스 생성
-- CREATE DATABASE DB_NAME;
CREATE DATABASE test;
