# ET Project

ET는 지식 그래프와 인지 과학을 결합한 개인화 학습/복습 플랫폼입니다.
사용자의 학습 상태를 추적하고, 최적의 복습 시점과 개인화된 학습 경로를 추천합니다.

---

## ⚙️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Backend** | Python 3.12, Django 5.0, Django REST Framework |
| **Database** | SQLite (Dev), Redis 5.0+ (Caching & Celery Broker) |
| **AI / ML** | PyTorch (CPU), Sentence Transformers, UMAP, Scikit-learn |
| **LLM** | OpenAI API (GPT-4o / GPT-3.5) |
| **Task Queue** | Celery, RabbitMQ/Redis |
| **DevOps** | Nginx, Gunicorn, Systemd |

---

## 🚀 Getting Started

프로젝트를 로컬 환경에서 실행하는 방법입니다.

### 1. Prerequisites (사전 준비)
*   **Python 3.10+** (3.12 권장)
*   **Redis** (Celery 비동기 작업용)
    ```bash
    # Mac (Homebrew)
    brew install redis
    brew services start redis
    ```

### 2. Installation (설치)

1. **Repository Clone**
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Virtual Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables (.env)**
   프로젝트 루트(`backend/`)에 `.env` 파일을 생성하고 아래 내용을 작성하세요.
   ```ini
   # Django
   SECRET_KEY=development_secret_key
   DEBUG=True
   ALLOWED_HOSTS=*

   # OpenAI (Required for Knowledge Extraction)
   OPENAI_API_KEY=sk-your-openai-api-key

   # Redis
   REDIS_URL=redis://localhost:6379/0
   ```

5. **Database Migration**
   ```bash
   python manage.py migrate
   ```

### 3. Running Locally (실행)

**1. Django Server**
```bash
python manage.py runserver
```

**2. Celery Worker (비동기 작업 처리)**
별도의 터미널에서 실행하세요.
```bash
celery -A backend worker --loglevel=info
```

---

## 🧠 지식 노드 생성 로직 (Knowledge Node Creation)

지식 노드 생성은 **비정형 데이터(텍스트/PDF)를 정형화된 지식 그래프 노드(Atomic Concepts)로 변환**하는 핵심 프로세스입니다.

### 1. Ingestion (데이터 섭취)
입력된 데이터(텍스트 또는 PDF)를 처리 가능한 단위로 변환합니다.
- **모듈**: `services.knowledge.ingestion.IngestionService`
- **프로세스**:
  1. **소스 판별**: 텍스트, PDF 파일 여부를 확인합니다.
  2. **전처리 (Cleaning)**:
     - PDF의 경우 `PyMuPDF`를 사용하여 텍스트를 추출합니다.
     - 헤더/푸터, 페이지 번호 등 불필요한 노이즈를 제거합니다.
     - 연속된 공백이나 불필요한 줄바꿈을 정규화합니다.
  3. **청킹 (Chunking)**:
     - `chunk_size` (기본 4000자) 단위로 텍스트를 분할합니다.
     - 문맥 유지를 위해 단락(`\n\n`) 단위로 우선 분할하고, 너무 큰 경우 문장 단위로 나눕니다.
     - 인접 청크 간 `chunk_overlap`(200자)을 두어 정보 손실을 방지합니다.

### 2. Extraction (지식 추출)
청크 단위 텍스트에서 핵심 개념(Node)을 추출합니다.
- **모듈**: `services.knowledge.extractor.extract_nodes`
- **LLM 활용**:
  - LLM에게 "지식 공학자" 페르소나를 부여하여 핵심 개념을 추출합니다.
  - **제약 조건**:
    - 독립적인 학습 단위여야 함.
    - 제목은 명사형으로 간결하게.
    - 너무 일반적이거나(예: "수학"), 너무 지엽적인(예: 변수명) 개념 제외.
- **출력 포맷**: JSON (`title`, `description`, `tags`)

### 3. Deduplication (중복 제거)
추출된 노드가 기존 지식 베이스에 이미 존재하는지 확인하고 병합합니다.
- **전처리**: 제목 정규화 (소문자 변환, 공백/특수문자 제거, 약어 확장 등).
  - 예: `ML` -> `machinelearning`, `Deep Learning` -> `deeplearning`
- **유사도 비교**:
  1. **정확한 매칭**: 정규화된 제목이 정확히 일치하는지 확인.
  2. **포함 관계**: 한 제목이 다른 제목을 포함하는 경우 (유사도 0.8 부여).
  3. **Jaccard 유사도**: 문자(Character) 단위의 집합 유사도를 계산.
- **임계값**: 유사도가 `0.8` 이상이면 중복으로 판단하여 생성을 건너뜁니다.

---

## 📡 API 명세 (API Specifications)

### 1. Knowledge API (`/api/v1/knowledge/`)
지식 그래프 데이터 관리 및 조회.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/ingest/` | 텍스트/PDF를 업로드하여 지식 노드로 변환합니다. |
| **GET** | `/nodes/` | 지식 노드 목록을 조회합니다. |
| **GET** | `/nodes/<pk>/` | 특정 지식 노드의 상세 정보를 조회합니다. |
| **GET** | `/edges/` | 노드 간의 엣지(관계) 목록을 조회합니다. |
| **GET** | `/clusters/` | 클러스터링된 노드 목록을 조회합니다. |
| **POST** | `/recommend/` | GNN/알고리즘 기반 맞춤 학습 노드를 추천합니다. |
| **GET** | `/quiz/set/` | 학습용 퀴즈 세트를 조회합니다. |
| **POST** | `/nodes/<node_id>/quiz/` | 특정 노드에 대한 퀴즈를 즉시 생성합니다. |

### 2. Analytics API (`/api/v1/analytics/`)
학습 데이터 분석, 벤치마크 테스트 및 복습 스케줄링.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/benchmark/initialize/` | 벤치마크 테스트 세션을 시작합니다. |
| **POST** | `/benchmark/submit/` | 벤치마크 테스트 결과(정답/오답)를 제출합니다. |
| **POST** | `/benchmark/analyze/` | 제출된 결과를 분석하여 리포트를 생성합니다. |
| **GET** | `/benchmark/status/<user_id>/` | 사용자의 벤치마크 진행 상태를 확인합니다. |
| **GET** | `/results/<user_id>/` | 사용자의 종합 분석 결과를 조회합니다. |
| **GET** | `/schedules/` | 현재 활성화된 복습 스케줄 목록을 조회합니다. |
| **POST** | `/schedules/` | 수동으로 복습 스케줄을 생성합니다. |
| **POST** | `/schedules/auto/` | 알고리즘에 기반하여 최적의 복습 스케줄을 자동 생성합니다. |
| **GET** | `/schedules/upcoming/` | 곧 다가오는 복습 일정을 조회합니다. |
| **POST** | `/schedules/<id>/complete/` | 특정 복습 스케줄 완료 처리. |
| **POST** | `/push/subscribe/` | Web Push 알림을 구독합니다. |
| **GET** | `/push/vapid-key/` | Web Push용 VAPID 공개키를 조회합니다. |

### 3. Auth API (`/api/v1/auth/`)
사용자 인증 및 계정 관리.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/register/` | 신규 회원 가입. |
| **POST** | `/login/` | 로그인 및 토큰 발급. |

### 4. Debug API (`/api/v1/debug/`)
개발 및 테스트용 유틸리티 (프로덕션 주의).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health/` | 시스템 상태 확인 (Health Check). |
| **GET** | `/stats/` | DB 데이터 통계 조회. |
| **GET** | `/data/<model_name>/` | 특정 모델의 데이터 조회. |
| **POST** | `/seed/` | 테스트용 더미 데이터를 삽입합니다. |
| **POST** | `/clear/` | DB의 모든 데이터를 삭제합니다. |
| **POST** | `/reset/` | DB 초기화 및 시드 데이터 재생성. |
| **POST** | `/benchmark/quick/` | 벤치마크 빠른 테스트 데이터 생성. |
| **POST** | `/rebuild-db/` | DB 재생성 (Drop tables & Migrate). |

### 5. Common / Visualization
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/api/v1/universe/` | 3D 유니버스 시각화를 위한 전체 데이터셋 조회. |
| **GET** | `/api/v1/tasks/<task_id>/` | 비동기 작업(Celery)의 상태를 조회합니다. |
