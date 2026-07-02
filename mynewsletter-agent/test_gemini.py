"""
Gemini API 단독 테스트 스크립트
실행: python test_gemini.py
"""
from dotenv import load_dotenv
from main import summarize_with_gemini, summarize, log

load_dotenv()

KOREAN_SAMPLE = {
    "subject": "[테스트] AI 뉴스레터 2024",
    "sender": "test@example.com",
    "body": """
안녕하세요, 이번 주 AI 뉴스레터입니다.

OpenAI가 새로운 GPT-5 모델을 발표했습니다. 이 모델은 기존 GPT-4 대비 추론 능력이 2배 향상되었으며,
멀티모달 기능도 크게 강화되었습니다. 특히 코드 생성과 수학 문제 해결 분야에서 두각을 나타냈습니다.

Google DeepMind는 AlphaFold 3를 공개하며 단백질 구조 예측 분야에서 또 한 번의 혁신을 이뤄냈습니다.
이번 버전은 DNA, RNA, 소분자까지 예측 범위를 확장했습니다.

국내에서는 카카오가 자체 LLM인 KoGPT 3.0을 출시했으며, 한국어 이해도에서 기존 모델 대비
30% 향상된 성능을 보인다고 밝혔습니다.
""",
}

ENGLISH_SAMPLE = {
    "subject": "The future of AI agents in enterprise software",
    "sender": "newsletter@techdigest.com",
    "body": """
This week, we're diving deep into how AI agents are reshaping enterprise workflows.

Microsoft announced Copilot Studio enhancements that allow businesses to build custom agents
integrated directly into their existing Azure infrastructure. Early adopters at Fortune 500 companies
report 40% reduction in manual data entry tasks.

Salesforce launched AgentForce, a platform enabling sales teams to deploy autonomous AI agents
that handle lead qualification, follow-up scheduling, and CRM updates without human intervention.
The system uses a multi-agent architecture where specialized agents collaborate to complete complex tasks.

OpenAI's operator-class models are seeing strong adoption in the legal and finance sectors,
where document review workflows that previously took days can now be completed in hours.
""",
}


def test_single(label, sample):
    print(f"\n{'='*50}")
    print(f"테스트: {label}")
    print(f"제목: {sample['subject']}")
    print("=" * 50)

    summary, model = summarize(sample["subject"], sample["body"])

    print(f"\n[사용 모델: {model}]")
    print(f"\n--- 요약 결과 ---\n{summary}")
    print("-" * 50)


if __name__ == "__main__":
    print("Gemini 요약 테스트 시작\n")

    test_single("한글 뉴스레터", KOREAN_SAMPLE)
    test_single("영문 뉴스레터", ENGLISH_SAMPLE)

    print("\n테스트 완료")
