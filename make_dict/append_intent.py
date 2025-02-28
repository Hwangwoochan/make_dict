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
    
    for i in subjects:
        store_word_in_csv(i, "./subject.csv", "Subject")

    for i in objects:
        store_word_in_csv(i, "./object.csv", "Object")

    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from korean_lemmatizer.soylemma import Lemmatizer
    from conjugator import KConjugator


    lemmatizer = Lemmatizer()
    result_mapping = {}  # key: intent, value: 중복 없이 합쳐진 결과 리스트

    for intent in intents:
        # 1차 분석: intent에서 'Verb' 태그를 가진 단어 추출
        analysis = lemmatizer.analyze(intent)
        first_pass = [word for tup in analysis for word, pos in tup if pos == 'Verb']
        
        # 2차 분석: 1차 결과의 각 단어에 대해 재분석하여 'Verb' 추출
        second_pass = []
        for word in first_pass:
            analysis2 = lemmatizer.analyze(word)
            new_verbs = [w for tup in analysis2 for w, pos in tup if pos == 'Verb']
            for w in new_verbs:
                if w not in second_pass:
                    second_pass.append(w)
        
        # 두 결과를 합치고 중복 제거 (순서를 유지)
        combined = []
        for w in first_pass + second_pass:
            if w not in combined:
                combined.append(w)
        
        result_mapping[intent] = combined

    print(result_mapping)
    
    
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
    
    
    new_json = {}

    # 각 key에 대해 synonyms 업데이트
    for key, commands in result_mapping.items():
        new_vvv = []
        for command in commands:
            kc = KConjugator(command)
            
            # 1. 평서문 (Declarative): 현재, 비격식, 두 가지 높임
            mood = KConjugator.M_DECLARATIVE
            tense = KConjugator.T_PRESENT  # 현재만 사용
            formality = KConjugator.F_INFORMAL
            for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                result = kc.conjugate(mood, tense, formality, honorific)
                if result not in new_vvv:
                    new_vvv.append(result)
                    
            # 2. 명령문 (Imperative)
            mood = KConjugator.M_IMPERATIVE
            for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                        KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in new_vvv:
                            new_vvv.append(result)
                        
            # 3. 청유문 (Propositive)
            mood = KConjugator.M_PROPOSITIVE
            for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
                        KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in new_vvv:
                            new_vvv.append(result)
            #4. custom
            mood = KConjugator.M_ADJECTIVAL
            for tense in [KConjugator.T_FUTURE]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in new_vvv:
                            new_vvv.append(result)
            #5 의문 + 조건/미레, 
            mood = KConjugator.M_INQUISITIVE
            for tense in [KConjugator.T_CONDITIONAL]:
                for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
                    for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
                        result = kc.conjugate(mood, tense, formality, honorific)
                        if result not in new_vvv:
                            new_vvv.append(result)
        
        # 분리된 자음 제거 함수
        def remove_isolated_consonants(text: str) -> str:
            return re.sub(r'[\u3131-\u314E]', '', text)
        
        results = [remove_isolated_consonants(word) for word in new_vvv]
        
         # 새로운 JSON에 해당 명령어를 키로 결과 추가
        # 만약 같은 명령어가 이미 존재한다면 결과를 합쳐 중복 없이 저장합니다.
     
        new_json[key] = results
        
        for i in results:
            store_word_in_csv(i, "./intent.csv", "Intent")
            
    print(new_json)
    
    with open("new_result_mapping.json", "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
    # # 업데이트된 딕셔너리를 다시 JSON 파일로 저장
    # with open("result_mapping.json", "w", encoding="utf-8") as f:
    #     json.dump(result_mapping, f, ensure_ascii=False, indent=4)


    # for key, values in result_mapping.items():
    #     new_vvv = []
    #     for value in values:
    #         kc = KConjugator(value)
            
    #         # 1. 평서문 (Declarative): 현재, 비격식만 사용, 두 가지 높임
    #         mood = KConjugator.M_DECLARATIVE
    #         tense = KConjugator.T_PRESENT  # 현재만 사용
    #         formality = KConjugator.F_INFORMAL
    #         for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
    #             result = kc.conjugate(mood, tense, formality, honorific)
    #             conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
    #             if result not in new_vvv:
    #                 new_vvv.append(result)
                
                
    #         # 2. 명령문 (Imperative): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
    #         mood = KConjugator.M_IMPERATIVE
    #         for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
    #                     KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
    #             for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
    #                 for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
    #                     result = kc.conjugate(mood, tense, formality, honorific)
    #                     conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
    #                     if result not in new_vvv:
    #                         new_vvv.append(result)
                    
                        
    #         # 3. 청유문 (Propositive): 모든 시제에 대해 (현재, 과거, 미래, 조건/미래(겠), 대과거)
    #         mood = KConjugator.M_PROPOSITIVE
    #         for tense in [KConjugator.T_PRESENT, KConjugator.T_PAST, KConjugator.T_FUTURE, 
    #                     KConjugator.T_CONDITIONAL, KConjugator.T_PLUPERFECT]:
    #             for formality in [KConjugator.F_INFORMAL, KConjugator.F_FORMAL]:
    #                 for honorific in [KConjugator.H_LOW, KConjugator.H_HIGH]:
    #                     result = kc.conjugate(mood, tense, formality, honorific)
    #                     conjugation_str = f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}"
    #                     if result not in new_vvv:
    #                         new_vvv.append(result)
              

    #     def remove_isolated_consonants(text: str) -> str:
    #     # 한글 완성 음절은 U+AC00 ~ U+D7A3에 있으므로,
    #     # 이 범위에 속하지 않는 분리된 자음(ㄱ~ㅎ)은 제거합니다.
    #         return re.sub(r'[\u3131-\u314E]', '', text)

    #     results = [remove_isolated_consonants(word) for word in new_vvv]
    #     # extend 리스트에는 중복 없이 모든 결과 문자열이 저장됩니다.
    #     print(key,results)
        
    
    
    # for i in results:
    #     store_word_in_csv(i, "./intent.csv", "Intent")
    
   


if __name__ == "__main__":
    
    # text = convert("올라", sep=' ')
    # print(text)
    json_to_csv_ver2()
    
    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    
    # converter = JSONToCSVConverter("object.json")
    # converter.run()
    # run_input_mode()
