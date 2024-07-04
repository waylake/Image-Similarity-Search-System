# Image Similarity Search System

이 프로젝트는 웹 스크래핑을 통해 상품 정보를 수집하고, 상품 이미지를 cosine 유사도기반 검색 기능을 제공하는 시스템입니다.

## 실행
- **Docker**
 
- **docker-compose**

## 프로젝트 실행 방법 

### 1. 프로젝트 클론 및 이동 


```sh
git clone https://github.com/waylake/Image-Similarity-Search-System
cd Image-Similarity-Search-System
```

### 2. Docker Compose로 컨테이너 빌드 및 실행 


```sh
docker-compose up --build
```

빌드를 시작하면, 대략 10 ~ 20분 사이에 테스트 준비가 완료됩니다.

작업순서는 다음과 같습니다.
1. 데이터 수집
2. 이미지에서 SIFT Feature 추출, 이 작업의 진행상황은 docker client 앱에서 컨테이너 로그에서 확인이 가능합니다(cli 에서 확인이 되지 않습니다.)
3. Elasticsearch 에 인덱싱

위의 작업이 끝나면 http://localhost:8090/docs 에서 유사 이미지 검색이 가능합니다.

## 사용된 기술 및 스택

### 백엔드

- **Python 3.10**

- **FastAPI** : 웹 API 프레임워크

- **Elasticsearch** : 데이터 저장 및 검색 엔진

- **OpenCV** : 이미지 처리

- **SIFT (Scale-Invariant Feature Transform)** : 이미지 특징 추출

- **MinHash** : 로컬리티 센시티브 해싱(LSH)을 위한 기술

### 데이터 수집

- **requests** : HTTP 요청 처리

- **BeautifulSoup** : HTML 파싱

### 병렬 처리

- **concurrent.futures** : 멀티스레딩 구현

### 이미지 처리

- **Pillow** : 이미지 처리 라이브러리

- **NumPy** : 수치 연산

### 기타

- **python-dotenv** : 환경 변수 관리

- **tqdm** : 진행 상황 시각화

- **logging** : 로깅 시스템

### 컨테이너화

- **Docker**

- **docker-compose**

## 프로젝트 플로우

### 데이터 수집

- `ItemCollector` 클래스를 사용하여 웹사이트에서 상품 정보를 스크래핑합니다.

- 30일치의 데이터를 병렬로 수집합니다.

### 이미지 처리 및 특징 추출

- `ImageProcessor` 클래스를 사용하여 수집된 이미지를 처리합니다.

- 이미지 크기 조정, 그레이스케일 변환, 노이즈 제거 등의 전처리를 수행합니다.

- SIFT 알고리즘을 사용하여 이미지 특징을 추출합니다.

### Feature hashing

- 추출된 특징을 MinHash를 사용하여 해싱합니다.

- 이는 효율적인 유사도 검색을 위한 준비 단계입니다.

### 데이터 인덱싱

- 처리된 데이터를 Elasticsearch에 인덱싱합니다.

- `ElasticsearchIndexer` 클래스가 이 작업을 담당합니다.

### API 서비스

- FastAPI를 사용하여 이미지 유사도 검색 API를 제공합니다.

- 사용자가 이미지 URL을 입력하면, 시스템은 유사한 이미지들을 검색하여 반환합니다.

### 검색 프로세스

- 입력 이미지에 대해 1-4 단계와 동일한 처리를 수행합니다.

- Elasticsearch의 `script_score` 쿼리를 사용하여 코사인 유사도 기반의 검색을 수행합니다.

### 초기화 및 업데이트

- `init_script.py`를 통해 시스템 초기화 및 데이터 업데이트를 수행할 수 있습니다.

- 필요에 따라 전체 데이터를 새로 수집하거나 기존 데이터를 사용할 수 있습니다.

