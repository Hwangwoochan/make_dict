import os
import json
import pandas as pd
from src.worker import convert
import re

###########################
# 기존 유틸 함수
###########################
def expand_dataframe_if_needed(df: pd.DataFrame, row_idx: int, col_idx: int) -> pd.DataFrame:
    needed_rows = row_idx + 1
    if needed_rows > df.shape[0]:
        add_rows = needed_rows - df.shape[0]
        new_data = [[None]*df.shape[1] for _ in range(add_rows)]
        df = pd.concat([df, pd.DataFrame(new_data, columns=df.columns)], ignore_index=True)

    needed_cols = col_idx + 1
    if needed_cols > df.shape[1]:
        add_cols = needed_cols - df.shape[1]
        new_col_names = range(df.shape[1], df.shape[1] + add_cols)
        for c in new_col_names:
            df[c] = None

    return df

def csv_contains_word(csv_path: str, word: str) -> bool:
    """
    CSV 파일에 'word'가 이미 존재하는지 검사.
    - 1행(인덱스=0)은 숫자정보이므로, 2행 이후부터 텍스트로 본다.
    - 만약 CSV가 없으면 False.
    """
    if not os.path.exists(csv_path):
        return False

    clean = pd.read_csv(csv_path, header=None)
    if clean.shape[0] <= 1:
        # 1행(인덱스=0)만 존재하면 실제 데이터가 없는 상태
        return False

    # 1행(인덱스=0)은 숫자 정보라 가정 -> 2행 이후가 실제 텍스트
    data_part = clean.iloc[1:].astype(str)
    # values를 1차원으로 펼침
    all_values = data_part.values.flatten()
    return word in all_values


def store_word_in_csv(word: str, csv_path: str, label: str):
    """
    공통 로직:
      1) 중복 체크
      2) convert()
      3) row/col 계산 & expand
      4) CSV 저장
    label은 'Subject', 'Intent', 'Option' 등
    """
    if csv_contains_word(csv_path, word):
        print(f"[{label} CSV] '{word}' 이미 존재 -> 저장 생략.")
        return
    
    text = convert(word, sep=' ')
    han = word
    length = text.count(" ") + 1
    
    if csv_contains_word(csv_path, han):
        print(f"[{label} CSV] '{han}' 이미 존재 -> 저장 생략.")
        return
    
    if not os.path.exists(csv_path):
        pd.DataFrame([[0]]).to_csv(csv_path, index=False, header=False, encoding="UTF-8-SIG")
    clean = pd.read_csv(csv_path, header=None)

    row0 = pd.to_numeric(clean.iloc[0], errors='coerce').fillna(0).astype(int)
    clean.iloc[0] = row0
    sub_lengths = clean.iloc[0].dropna().values.astype(int)

    needed_len_idx = 2 * length - 3
    if needed_len_idx >= len(sub_lengths):
        extension_size = needed_len_idx - len(sub_lengths) + 1
        sub_lengths = list(sub_lengths) + [len(sub_lengths)+1]*extension_size

    row_index = 2 * length - 4
    col_index = sub_lengths[2 * length - 3]

    clean = expand_dataframe_if_needed(clean, col_index + 1, row_index)
    clean = expand_dataframe_if_needed(clean, col_index + 1, row_index + 1)

    clean.iloc[col_index+1, row_index] = text
    clean.iloc[col_index+1, row_index+1] = han

    current_val = clean.iloc[0, row_index+1]
    if pd.isna(current_val):
        current_val = 0
    clean.iloc[0, row_index+1] = int(current_val) + 1

    row0 = pd.to_numeric(clean.iloc[0], errors='coerce').fillna(0).astype(int)
    clean.iloc[0] = row0

    clean.to_csv(csv_path, encoding="UTF-8-SIG", index=False, header=False)
    print(f"[{label} CSV] '{word}' -> row={col_index+1}, col={row_index} 저장 완료.")


###########################
# 목적어(object) JSON 저장
###########################
def store_object_in_json(subject: str, intent: str, obj: str = None, json_path="object.json"):
    """
    (subject, intent)에 목적어 obj를 추가. 
    - obj=None 이면 '목적어 리스트'만 생성하고 아무것도 넣지 않는다. (목적어 없는 관계)
    - 이미 등록된 목적어면 추가 X
    JSON 예:
    {
      "에어컨": {
        "켜": ["온도", "세기"],
        "꺼": []
      },
      "컴퓨터": {
        "켜": []
      }
    }
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if subject not in data:
        data[subject] = {}
    if intent not in data[subject]:
        data[subject][intent] = []

    if obj is not None:
        # 중복 방지
        if obj not in data[subject][intent]:
            data[subject][intent].append(obj)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    msg_obj = f"object='{obj}'" if obj else "object=None"
    print(f"[Object JSON] subject='{subject}', intent='{intent}', {msg_obj} 저장 완료.")

def json_to_csv (json_path="object.json"):
    if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)


    # 중복 없이 단어들을 저장할 리스트 초기화
    subjects = []  # 상위 키: 예) 에어컨, 컴퓨터, ...
    intents = []   # 내부 키: 예) 꺼줘, 켜줘, ...
    objects = []   # 내부 값: 예) 삼십도, 온도, ...

    for subject, inner in data.items():
        if subject not in subjects:
            subjects.append(subject)
        for intent, obj_list in inner.items():
            if intent not in intents:
                intents.append(intent)
            # 리스트가 비어있을 경우에도 처리 가능 (여기서는 object를 추가하지 않음)
            for obj in obj_list:
                if obj not in objects:
                    objects.append(obj)
    print("원본", intents)
    
    # for i in subjects:
    #     store_word_in_csv(i, "./subject.csv", "Subject")

    # for i in objects:
    #     store_word_in_csv(i, "./object.csv", "Object")

    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from korean_lemmatizer.soylemma import Lemmatizer
    from conjugator import KConjugator


    verb_words = []  # 반복문 밖에서 리스트 초기화
    
    lemmatizer = Lemmatizer()
    for i in intents:
        data = lemmatizer.analyze(i)
        verb_words.extend([word for tup in data for word, pos in tup if pos == 'Verb'])
   
    print("1차",verb_words)
    extend = []
    for i in verb_words:
        data = lemmatizer.analyze(i)
        # print(data)
        # lemmatizer.analyze의 결과에서 'Verb' 태그를 가진 단어들을 추출
        new_verbs = [word for tup in data for word, pos in tup if pos == 'Verb']
        # extend 리스트에 중복 없이 추가
        for word in new_verbs:
            if word not in extend:
                extend.append(word)

    print("2차",extend)
    
            
    # 각 상수에 대한 이름 매핑 (출력용)
    moodNames = {
        KConjugator.M_DECLARATIVE: "평서문",
        KConjugator.M_INQUISITIVE: "의문문",
        KConjugator.M_IMPERATIVE: "명령문",
        KConjugator.M_PROPOSITIVE: "청유문",
        KConjugator.M_NOMINAL: "명사형",
        KConjugator.M_ADJECTIVAL: "관형사형"
    }
    tenseNames = {
        KConjugator.T_PRESENT: "현재",
        KConjugator.T_PAST: "과거",
        KConjugator.T_FUTURE: "미래",
        KConjugator.T_CONDITIONAL: "조건/미래(겠)",
        KConjugator.T_PLUPERFECT: "대과거"
    }
    formalNames = {
        KConjugator.F_INFORMAL: "비격식",
        KConjugator.F_FORMAL: "격식"
    }
    honorificNames = {
        KConjugator.H_LOW: "낮은 높임",
        KConjugator.H_HIGH: "높은 높임"
    }
    
    new_vvv = []

    for i in extend:
        kc = KConjugator(i)
        
        # 1. 평서문 (Declarative): 현재, 비격식만 사용, 두 가지 높임
        mood = KConjugator.M_DECLARATIVE
        tense = KConjugator.T_PRESENT  # 현재만 사용
        formality = KConjugator.F_INFORMAL
        for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
            result = kc.conjugate(mood, tense, formality, honorific)
            conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
            if result not in new_vvv:
                new_vvv.append(result)
            
            
        # 2. 명령문 (Imperative): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
        mood = KConjugator.M_IMPERATIVE
        for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                    KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
            for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                    result = kc.conjugate(mood, tense, formality, honorific)
                    conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
                    if result not in new_vvv:
                        new_vvv.append(result)
                   
                    
        # 3. 청유문 (Propositive): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
        mood = KConjugator.M_PROPOSITIVE
        for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                    KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
            for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                    result = kc.conjugate(mood, tense, formality, honorific)
                    conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
                    if result not in new_vvv:
                        new_vvv.append(result)
              

    def remove_isolated_consonants(text: str) -> str:
    # 한글 완성 음절은 U+AC00 ~ U+D7A3에 있으므로,
    # 이 범위에 속하지 않는 분리된 자음(ㄱ~ㅎ)은 제거합니다.
        return re.sub(r'[\u3131-\u314E]', '', text)

    results = [remove_isolated_consonants(word) for word in new_vvv]
    # extend 리스트에는 중복 없이 모든 결과 문자열이 저장됩니다.
    print(results)
    
    
    
    for i in results:
        store_word_in_csv(i, "./intent.csv", "Intent")
    
   


###########################
# 메인 실행부
###########################
def run_input_mode():
    """
    사용자 입력:
    - a b c (3단어) : 주어 a, 목적어 b, 의도 c
    - a c   (2단어) : 주어 a, 의도 c
      => 목적어는 없음
    종료: q

    subject2.csv, intent2.csv:
        - 중복이면 저장 생략
    object.json:
        - 중복이면 저장 안 함
    """
    print("=== Input Mode ===")
    print("형식:")
    print("  1) 'a b c' (예: '에어컨 온도 맞춰')")
    print("  2) 'a c'   (예: '에어컨 꺼')")
    print("종료: q")

    while True:
        usr_input = input("\nEnter (a b c) or (a c), or 'q' to quit: ").strip()
        if usr_input.lower() == 'q':
            print("종료합니다.")
            break

        tokens = usr_input.split()
        if len(tokens) == 2:
            # a c
            a, c = tokens
            store_word_in_csv(a, "./subject.csv", "Subject")
            store_word_in_csv(c, "./intent.csv", "Intent")
            store_object_in_json(a, c, None, "object.json")

        elif len(tokens) == 3:
            # a b c
            a, b, c = tokens
            store_word_in_csv(a, "./subject.csv", "Subject")
            store_word_in_csv(c, "./intent.csv", "Intent")
            store_word_in_csv(b, "./object.csv", "Object")
            store_object_in_json(a, c, b, "object.json")

        else:
            print("[오류] 2단어(a c) 또는 3단어(a b c)로 입력해주세요.")




def json_to_csv_ver2 (json_path="object.json"):
    if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)


    # 중복 없이 단어들을 저장할 리스트 초기화
    subjects = []  # 상위 키: 예) 에어컨, 컴퓨터, ...
    intents = []   # 내부 키: 예) 꺼줘, 켜줘, ...
    objects = []   # 내부 값: 예) 삼십도, 온도, ...

    for subject, inner in data.items():
        if subject not in subjects:
            subjects.append(subject)
        for intent, obj_list in inner.items():
            if intent not in intents:
                intents.append(intent)
            # 리스트가 비어있을 경우에도 처리 가능 (여기서는 object를 추가하지 않음)
            for obj in obj_list:
                if obj not in objects:
                    objects.append(obj)
    print("원본", intents)
    
    # for i in subjects:
    #     store_word_in_csv(i, "./subject.csv", "Subject")

    # for i in objects:
    #     store_word_in_csv(i, "./object.csv", "Object")

    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from korean_lemmatizer.soylemma import Lemmatizer
    from conjugator import KConjugator


    verb_words = []  # 반복문 밖에서 리스트 초기화
    
    lemmatizer = Lemmatizer()
    for i in intents:
        data = lemmatizer.analyze(i)
        verb_words.extend([word for tup in data for word, pos in tup if pos == 'Verb'])
   
        print("1차",verb_words)
        extend = []
        for i in verb_words:
            data = lemmatizer.analyze(i)
            # print(data)
            # lemmatizer.analyze의 결과에서 'Verb' 태그를 가진 단어들을 추출
            new_verbs = [word for tup in data for word, pos in tup if pos == 'Verb']
            # extend 리스트에 중복 없이 추가
            for word in new_verbs:
                if word not in extend:
                    extend.append(word)

        print("2차",extend)
    
            
    # 각 상수에 대한 이름 매핑 (출력용)
    moodNames = {
        KConjugator.M_DECLARATIVE: "평서문",
        KConjugator.M_INQUISITIVE: "의문문",
        KConjugator.M_IMPERATIVE: "명령문",
        KConjugator.M_PROPOSITIVE: "청유문",
        KConjugator.M_NOMINAL: "명사형",
        KConjugator.M_ADJECTIVAL: "관형사형"
    }
    tenseNames = {
        KConjugator.T_PRESENT: "현재",
        KConjugator.T_PAST: "과거",
        KConjugator.T_FUTURE: "미래",
        KConjugator.T_CONDITIONAL: "조건/미래(겠)",
        KConjugator.T_PLUPERFECT: "대과거"
    }
    formalNames = {
        KConjugator.F_INFORMAL: "비격식",
        KConjugator.F_FORMAL: "격식"
    }
    honorificNames = {
        KConjugator.H_LOW: "낮은 높임",
        KConjugator.H_HIGH: "높은 높임"
    }
    
    new_vvv = []

    for i in extend:
        kc = KConjugator(i)
        
        # 1. 평서문 (Declarative): 현재, 비격식만 사용, 두 가지 높임
        mood = KConjugator.M_DECLARATIVE
        tense = KConjugator.T_PRESENT  # 현재만 사용
        formality = KConjugator.F_INFORMAL
        for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
            result = kc.conjugate(mood, tense, formality, honorific)
            conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
            if result not in new_vvv:
                new_vvv.append(result)
            
            
        # 2. 명령문 (Imperative): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
        mood = KConjugator.M_IMPERATIVE
        for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                    KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
            for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                    result = kc.conjugate(mood, tense, formality, honorific)
                    conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
                    if result not in new_vvv:
                        new_vvv.append(result)
                   
                    
        # 3. 청유문 (Propositive): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
        mood = KConjugator.M_PROPOSITIVE
        for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                    KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
            for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                    result = kc.conjugate(mood, tense, formality, honorific)
                    conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
                    if result not in new_vvv:
                        new_vvv.append(result)
              

    def remove_isolated_consonants(text: str) -> str:
    # 한글 완성 음절은 U+AC00 ~ U+D7A3에 있으므로,
    # 이 범위에 속하지 않는 분리된 자음(ㄱ~ㅎ)은 제거합니다.
        return re.sub(r'[\u3131-\u314E]', '', text)

    results = [remove_isolated_consonants(word) for word in new_vvv]
    # extend 리스트에는 중복 없이 모든 결과 문자열이 저장됩니다.
    print(results)
    
    
    
    for i in results:
        store_word_in_csv(i, "./intent.csv", "Intent")
    
   





import os
import sys
import json
import re

# 외부에서 제공하는 함수라고 가정 (예: CSV에 저장하는 함수)
# from some_module import store_word_in_csv

class JSONToCSVConverter:
    def __init__(self, json_path="object.json"):
        self.json_path = json_path
        self.data = {}
        self.subjects = []
        self.intents = []
        self.objects = []
        self.verb_words = []
        self.conjugated_intents = []
        self.lemmatizer = None

    def load_json(self) -> bool:
        """JSON 파일을 로드합니다."""
        if not os.path.exists(self.json_path):
            print(f"JSON 파일 '{self.json_path}'을(를) 찾을 수 없습니다.")
            return False
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return True

    def extract_unique_elements(self):
        """JSON 데이터에서 subject, intent, object를 중복 없이 추출합니다."""
        for subject, inner in self.data.items():
            if subject not in self.subjects:
                self.subjects.append(subject)
            for intent, obj_list in inner.items():
                if intent not in self.intents:
                    self.intents.append(intent)
                for obj in obj_list:
                    if obj not in self.objects:
                        self.objects.append(obj)

    @staticmethod
    def remove_isolated_consonants(text: str) -> str:
        """
        한글 완성 음절(가-힣) 외의 분리된 자음(ㄱ~ㅎ)을 제거합니다.
        예를 들어, "올ㄹ라라" -> "올라라"로 변환됩니다.
        """
        return re.sub(r'[\u3131-\u314E]', '', text)

    def init_lemmatizer(self):
        """Lemmatizer를 초기화합니다."""
        from korean_lemmatizer.soylemma import Lemmatizer
        self.lemmatizer = Lemmatizer()

    def analyze_intents(self):
        """
        intents 목록을 분석하여 동사(Verb) 형태의 단어들을 추출합니다.
        첫 번째 pass: 각 intent에서 동사를 추출하여 임시 리스트(verb_words_first)에 저장합니다.
        두 번째 pass: 첫 번째 결과의 각 단어를 재분석하여 'Verb' 태그를 가진 단어를 추출한 후,
                중복 없이 최종 동사 목록(verb_words_final)에 저장합니다.
        최종 결과는 self.verb_words에 저장됩니다.
        """
        # 첫 번째 pass: intents에서 동사 추출
        verb_words_first = []
        for intent in self.intents:
            analysis = self.lemmatizer.analyze(intent)
            extracted = [word for tup in analysis for word, pos in tup if pos == 'Verb']
            for word in extracted:
                if word not in verb_words_first:
                    verb_words_first.append(word)
        print("1차 결과:", verb_words_first)
        
        # 두 번째 pass: 첫 번째 결과의 동사들을 재분석하여 최종 동사 추출
        verb_words_final = []
        for word in verb_words_first:
            analysis = self.lemmatizer.analyze(word)
            new_verbs = [w for tup in analysis for w, pos in tup if pos == 'Verb']
            for w in new_verbs:
                if w not in verb_words_final:
                    verb_words_final.append(w)
        print("2차 결과:", verb_words_final)
        
        # 최종 결과 저장
        self.verb_words = verb_words_final


    def generate_conjugations(self):
        """
        추출된 동사들에 대해 KConjugator를 이용하여
        평서문, 명령문, 청유문 형태의 활용형을 생성하고,
        각 결과에서 분리된 자음을 제거한 후 중복 없이 self.conjugated_intents에 저장합니다.
        """
        from conjugator import KConjugator
        
        conjugations = []
        for verb in self.verb_words:
            kc = KConjugator(verb)
            # 1. 평서문: 현재, 비격식, 낮은/높은 높임
            mood = KConjugator.M_DECLARATIVE
            tense = KConjugator.T_PRESENT
            formality = KConjugator.F_INFORMAL
            for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                result = kc.conjugate(mood, tense, formality, honorific)
                if result not in conjugations:
                    conjugations.append(result)
            # 2. 명령문: 모든 시제, 격식/비격식, 낮은/높은 높임
            mood = KConjugator.M_IMPERATIVE
            for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE,
                          KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in conjugations:
                            conjugations.append(result)
            # 3. 청유문: 모든 시제, 격식/비격식, 낮은/높은 높임
            mood = KConjugator.M_PROPOSITIVE
            for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE,
                          KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in conjugations:
                            conjugations.append(result)
        # 분리 자음 제거 후 저장
        self.conjugated_intents = [self.remove_isolated_consonants(word) for word in conjugations]

    def store_to_csv(self):
        """생성된 결과를 CSV 파일에 저장합니다."""
        for intent in self.conjugated_intents:
            store_word_in_csv(intent, "./intent.csv", "Intent")
        # subjects나 objects도 필요 시 저장 가능
        # for subject in self.subjects:
        #     store_word_in_csv(subject, "./subject.csv", "Subject")
        # for obj in self.objects:
        #     store_word_in_csv(obj, "./object.csv", "Object")

    def run(self):
        """전체 프로세스를 실행합니다."""
        if not self.load_json():
            return
        self.extract_unique_elements()
        self.init_lemmatizer()
        self.analyze_intents()
        self.generate_conjugations()
        # self.store_to_csv()
        print("CSV 변환 완료.")




if __name__ == "__main__":
    
    # text = convert("올라", sep=' ')
    # print(text)
    # json_to_csv_ver2()
    
    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    
    # converter = JSONToCSVConverter("object.json")
    # converter.run()
    run_input_mode()
