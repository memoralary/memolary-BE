"""
Cognitive Benchmark Simulation - 망각 곡선 추정 시뮬레이션

이 스크립트는 초기 인지 실험의 전체 흐름을 시뮬레이션합니다:
1. 테스트 데이터 생성 (CS + 경상도 사투리 노드)
2. 사용자 생성 및 벤치마크 초기화
3. T0~T3 테스트 결과 시뮬레이션
4. 망각 곡선 분석 및 리포트 생성
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

import django
django.setup()

from django.utils import timezone

from knowledge.models import KnowledgeNode, TrackType
from analytics.models import User, TestSession, TestResult, TimePoint, TestType
from services.cognitive.benchmark import (
    CognitiveBenchmark,
    BenchmarkReporter,
    Domain,
    CS_TAGS,
    DIALECT_TAGS,
)


# =============================================================================
# 테스트 데이터 생성
# =============================================================================

def create_test_nodes():
    """CS 및 경상도 사투리 테스트 노드 생성"""
    
    # CS 도메인 노드
    cs_nodes_data = [
        {"title": "해시 테이블", "description": "키-값 쌍을 저장하는 자료구조로, O(1) 평균 시간복잡도로 검색 가능", "tags": ["자료구조", "algorithm"]},
        {"title": "B-Tree", "description": "데이터베이스 인덱스에 사용되는 균형 트리 자료구조", "tags": ["데이터베이스", "자료구조"]},
        {"title": "세마포어", "description": "동시성 제어를 위한 동기화 도구로 P와 V 연산을 사용", "tags": ["운영체제", "동시성"]},
        {"title": "TCP 3-way Handshake", "description": "TCP 연결 수립을 위한 SYN-SYN/ACK-ACK 과정", "tags": ["네트워크", "protocol"]},
        {"title": "다익스트라 알고리즘", "description": "가중치가 있는 그래프에서 최단 경로를 찾는 그리디 알고리즘", "tags": ["알고리즘", "graph"]},
        {"title": "가상 메모리", "description": "물리 메모리보다 큰 주소 공간을 제공하는 메모리 관리 기법", "tags": ["운영체제", "memory"]},
        {"title": "정규화 (1NF, 2NF, 3NF)", "description": "데이터 중복을 제거하고 무결성을 보장하는 데이터베이스 설계 기법", "tags": ["데이터베이스", "설계"]},
        {"title": "HTTP/2 멀티플렉싱", "description": "단일 TCP 연결에서 여러 요청/응답을 동시에 처리하는 기술", "tags": ["네트워크", "web"]},
        {"title": "힙 정렬", "description": "완전 이진 트리의 힙 속성을 이용한 O(n log n) 정렬 알고리즘", "tags": ["알고리즘", "sorting"]},
        {"title": "페이지 교체 알고리즘", "description": "LRU, FIFO 등 메모리 부족 시 교체할 페이지를 선택하는 전략", "tags": ["운영체제", "memory"]},
    ]
    
    # 경상도 사투리 노드
    dialect_nodes_data = [
        {"title": "가가 가가?", "description": "그 사람이 그 사람인가? (가: 그 사람, 가가: 그 사람이)", "tags": ["경상도_사투리", "대명사"]},
        {"title": "니 머하노?", "description": "너 뭐하니? (무엇을 하고 있는지 묻는 표현)", "tags": ["경상도_사투리", "의문문"]},
        {"title": "쪼매", "description": "조금, 약간 (소량을 나타내는 부사)", "tags": ["경상도_사투리", "부사"]},
        {"title": "미따", "description": "밉다, 싫다 (부정적 감정 표현)", "tags": ["경상도_사투리", "형용사"]},
        {"title": "와따", "description": "대단하다, 멋지다 (감탄 표현)", "tags": ["경상도_사투리", "감탄사"]},
        {"title": "무꼬", "description": "뭐하고? (줄임 표현)", "tags": ["경상도_사투리", "의문문"]},
        {"title": "카네", "description": "~라고 하네 (전달 표현)", "tags": ["경상도_사투리", "종결어미"]},
        {"title": "안카나", "description": "~하지 않나, 그렇지 않니? (확인 의문)", "tags": ["경상도_사투리", "종결어미"]},
        {"title": "마이", "description": "많이 (양을 나타내는 부사)", "tags": ["경상도_사투리", "부사"]},
        {"title": "가불다", "description": "날씨가 춥다 (추운 느낌의 형용사)", "tags": ["경상도_사투리", "형용사"]},
    ]
    
    created_cs = []
    created_dialect = []
    
    # CS 노드 생성
    for data in cs_nodes_data:
        node, created = KnowledgeNode.objects.get_or_create(
            title=data["title"],
            defaults={
                "description": data["description"],
                "tags": data["tags"],
                "track_type": TrackType.TRACK_A,
            }
        )
        created_cs.append(node)
        if created:
            print(f"  ✓ CS 노드 생성: {data['title']}")
    
    # 사투리 노드 생성
    for data in dialect_nodes_data:
        node, created = KnowledgeNode.objects.get_or_create(
            title=data["title"],
            defaults={
                "description": data["description"],
                "tags": data["tags"],
                "track_type": TrackType.TRACK_B,
            }
        )
        created_dialect.append(node)
        if created:
            print(f"  ✓ 사투리 노드 생성: {data['title']}")
    
    return created_cs, created_dialect


# =============================================================================
# 테스트 결과 시뮬레이션
# =============================================================================

def simulate_test_results(user, sessions, cs_nodes, dialect_nodes):
    """
    시뮬레이션된 테스트 결과 생성
    
    인지 공학적으로 현실적인 패턴을 시뮬레이션:
    - CS 도메인: 높은 초기 정답률, 느린 망각
    - 사투리 도메인: 낮은 초기 정답률, 빠른 망각
    - 시간이 지날수록 정답률 감소 (에빙하우스 곡선)
    """
    
    # 시점별 기억 유지율 시뮬레이션 파라미터
    # CS: k=0.2 (느린 망각), Dialect: k=0.5 (빠른 망각)
    retention_params = {
        # time_point: (cs_base_accuracy, dialect_base_accuracy, rt_range)
        'T0': (0.85, 0.60, (1500, 3500)),   # 즉시: 높은 정답률
        'T1': (0.78, 0.48, (2000, 4000)),   # 10분 후
        'T2': (0.72, 0.38, (2500, 4500)),   # 1시간 후
        'T3': (0.65, 0.28, (3000, 5000)),   # 24시간 후
    }
    
    results = []
    
    for time_point, session in sessions.items():
        params = retention_params.get(time_point, (0.5, 0.3, (2000, 4000)))
        cs_base, dialect_base, rt_range = params
        
        # 테스트 유형 결정
        if time_point == 'T0':
            test_types = [TestType.A1_BOTTOM_UP, TestType.A2_TOP_DOWN]
        else:
            test_types = [TestType.B_RECALL]
        
        print(f"\n[{time_point}] 시뮬레이션 진행...")
        
        # CS 도메인 결과
        for node in cs_nodes:
            test_type = random.choice(test_types)
            
            # 정답 여부 (확률적)
            is_correct = random.random() < cs_base
            
            # 확신도 (정답이면 높게, 오답이면 낮게 but 과신 경향 있음)
            if is_correct:
                confidence = random.uniform(0.65, 0.95)
            else:
                confidence = random.uniform(0.40, 0.75)  # 과신 경향
            
            # 반응 시간
            rt = random.randint(*rt_range)
            if is_correct:
                rt = int(rt * 0.8)  # 정답이면 더 빠름
            
            result = TestResult.objects.create(
                session=session,
                node=node,
                is_correct=is_correct,
                confidence_score=round(confidence, 2),
                response_time_ms=rt,
                test_type=test_type
            )
            results.append(result)
        
        # 사투리 도메인 결과
        for node in dialect_nodes:
            test_type = random.choice(test_types)
            
            # 정답 여부 (더 낮은 확률)
            is_correct = random.random() < dialect_base
            
            # 확신도 (신규 도메인에서는 더 신중함)
            if is_correct:
                confidence = random.uniform(0.50, 0.85)
            else:
                confidence = random.uniform(0.25, 0.55)  # 신중 경향
            
            # 반응 시간 (더 느림)
            rt = random.randint(*rt_range)
            rt = int(rt * 1.2)  # 사투리는 더 오래 걸림
            
            result = TestResult.objects.create(
                session=session,
                node=node,
                is_correct=is_correct,
                confidence_score=round(confidence, 2),
                response_time_ms=rt,
                test_type=test_type
            )
            results.append(result)
        
        print(f"    ✓ {len(cs_nodes) + len(dialect_nodes)}개 결과 생성")
    
    return results


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    print("=" * 70)
    print("🧠 초기 인지 실험 시뮬레이션")
    print("=" * 70)
    print("""
    목적: 익숙한 도메인(CS)과 익숙하지 않은 도메인(경상도 사투리)에서
          기억 형성 및 붕괴 속도의 차이를 측정하여 개인화된 망각 곡선 추정
    """)
    
    # 1. 테스트 노드 생성
    print("\n[1] 테스트 노드 생성")
    print("-" * 50)
    cs_nodes, dialect_nodes = create_test_nodes()
    print(f"\n    CS 노드: {len(cs_nodes)}개")
    print(f"    사투리 노드: {len(dialect_nodes)}개")
    
    # 2. 테스트 사용자 생성
    print("\n[2] 테스트 사용자 생성")
    print("-" * 50)
    user, created = User.objects.get_or_create(
        username="benchmark_user_001",
        defaults={"alpha_user": 1.0, "base_forgetting_k": 0.5}
    )
    print(f"    사용자: {user.username} {'(신규)' if created else '(기존)'}")
    
    # 3. 벤치마크 초기화 (세션 생성)
    print("\n[3] 벤치마크 세션 생성")
    print("-" * 50)
    benchmark = CognitiveBenchmark(nodes_per_domain=10)
    
    sessions = {}
    start_time = timezone.now()
    
    for tp, minutes in [('T0', 0), ('T1', 10), ('T2', 60), ('T3', 1440)]:
        session, _ = TestSession.objects.get_or_create(
            user=user,
            time_point=tp,
            defaults={"scheduled_at": start_time + timedelta(minutes=minutes)}
        )
        sessions[tp] = session
        print(f"    {tp}: {session.get_time_point_display()}")
    
    # 4. 테스트 결과 시뮬레이션
    print("\n[4] 테스트 결과 시뮬레이션")
    print("-" * 50)
    
    # 기존 결과 삭제 (재실행 시)
    TestResult.objects.filter(session__user=user).delete()
    
    results = simulate_test_results(user, sessions, cs_nodes, dialect_nodes)
    print(f"\n    총 {len(results)}개 테스트 결과 생성")
    
    # 5. 분석 실행
    print("\n[5] 망각 곡선 분석")
    print("-" * 50)
    
    cs_node_ids = [str(n.id) for n in cs_nodes]
    dialect_node_ids = [str(n.id) for n in dialect_nodes]
    
    analysis_result = benchmark.analyze_results(
        user_id=str(user.id),
        cs_node_ids=cs_node_ids,
        dialect_node_ids=dialect_node_ids
    )
    
    # 6. 리포트 생성
    print("\n[6] 분석 리포트 생성")
    print("-" * 50)
    
    reporter = BenchmarkReporter()
    report = reporter.generate_report(analysis_result)
    
    # 7. 결과 출력
    print("\n" + "=" * 70)
    print("📊 분석 결과")
    print("=" * 70)
    
    summary = report['summary']
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │ 기초 망각 상수 (base_k)    : {summary['base_forgetting_k']:.4f}                          │
    │ CS 도메인 망각 기울기 (k_cs): {summary['k_cs']:.4f}                          │
    │ 사투리 도메인 망각 기울기   : {summary['k_dialect']:.4f}                          │
    │ 도메인 망각 비율            : {summary['domain_ratio']:.2f} (사투리/CS)               │
    │ 평균 착각 지수              : {summary['overall_illusion']:.3f}                           │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n📈 시점별 정답률 변화")
    print("-" * 50)
    temporal = report['temporal_comparison']
    print(f"    {'시점':^6} │ {'CS 정답률':^12} │ {'사투리 정답률':^12} │ {'차이':^8}")
    print("    " + "─" * 48)
    for tp in ['T0', 'T1', 'T2', 'T3']:
        data = temporal[tp]
        cs_acc = f"{data['cs_accuracy']:.1%}" if data['cs_accuracy'] else "N/A"
        dial_acc = f"{data['dialect_accuracy']:.1%}" if data['dialect_accuracy'] else "N/A"
        diff = f"{data['accuracy_difference']:+.1%}" if data['accuracy_difference'] else "N/A"
        print(f"    {tp:^6} │ {cs_acc:^12} │ {dial_acc:^12} │ {diff:^8}")
    
    print("\n⏱️ 시점별 평균 반응 시간 (ms)")
    print("-" * 50)
    print(f"    {'시점':^6} │ {'CS':^12} │ {'사투리':^12}")
    print("    " + "─" * 36)
    for tp in ['T0', 'T1', 'T2', 'T3']:
        data = temporal[tp]
        cs_rt = f"{data['cs_rt_ms']:.0f}" if data['cs_rt_ms'] else "N/A"
        dial_rt = f"{data['dialect_rt_ms']:.0f}" if data['dialect_rt_ms'] else "N/A"
        print(f"    {tp:^6} │ {cs_rt:^12} │ {dial_rt:^12}")
    
    print("\n🔍 인지 특성 해석")
    print("-" * 50)
    interpretation = report['cognitive_interpretation']
    print(f"\n    [망각 패턴]")
    print(f"    {interpretation['forgetting_pattern']}")
    print(f"\n    [인코딩 패턴]")
    print(f"    {interpretation['encoding_pattern']}")
    print(f"\n    [메타인지 패턴]")
    print(f"    {interpretation['metacognition_pattern']}")
    print(f"\n    [도메인 전이]")
    print(f"    {interpretation['domain_transfer']}")
    
    print("\n💡 권장사항")
    print("-" * 50)
    print(f"    {summary['recommendation']}")
    
    # JSON 리포트 저장
    report_path = '/Users/myeongsung/ET/backend/benchmark_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 전체 리포트 저장: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 시뮬레이션 완료")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()
