# -*- coding: utf-8 -*-
import re

class KConjugator:
    # 한글 자모 배열
    u_cho = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
    u_jung = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
    u_jong = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

    # 불규칙 동사 예외 배열들
    reg_b = ['털썩이잡', '넘겨잡', '우접', '입', '맞접', '문잡', '다잡', '까뒤집', '배좁', '목잡', '끄집', '잡', '옴켜잡', '검잡', '되순라잡', '내씹', '모집', '따잡', '엇잡', '까집', '겹집', '줄통뽑', '버르집', '지르잡', '추켜잡', '업', '되술래잡', '되접', '좁디좁', '더위잡', '말씹', '내뽑', '집', '걸머잡', '휘어잡', '꿰입', '황잡', '에굽', '내굽', '따라잡', '맞뒤집', '둘러업', '늘잡', '끄잡', '우그려잡', '어줍', '언걸입', '들이곱', '껴잡', '곱 접', '훔켜잡', '늦추잡', '갈아입', '친좁', '희짜뽑', '마음잡', '개미잡', '옴씹', '치잡', '그러잡', '움켜잡', '씹', '비집', '꼽', '살잡', '죄입', '졸잡', '가려잡', '뽑', '걷어잡', '헐잡', '돌라입', '덧잡', '얕잡', '낫잡', '부여잡', '맞붙잡', '걸입', '주름잡', '걷어입', '빌미잡', '개잡', '겉잡', '안쫑잡', '좁', '힘입', '걷잡', '바르집', '감씹', '짓씹', '손잡', '포집', '붙잡', '낮잡', '책잡', '곱잡', '흉잡', '뒤집', '땡잡', '어림잡', '덧껴입', '수줍', '뒤잡', '꼬집', '예굽', '덮쳐잡', '헛잡', '되씹', '낮추잡', '날파람잡', '틀어잡', '헤집', '남의달잡', '바로잡', '흠잡', '파잡', '얼추잡', '손꼽', '접', '차려입', '골라잡', '거머잡', '후려잡', '머줍', '넉장뽑', '사로잡', '덧입', '껴입', '얼입', '우집', '설잡', '늦잡', '비좁', '고르잡', '때려잡', '떼집', '되잡', '홈켜잡', '내곱', '곱씹', '빼입', '들이굽', '새잡', '이르집', '떨쳐입']
    reg_s = ['내솟', '빗', '드솟', '비웃', '뺏', '샘솟', '벗', '들이웃', '솟', '되뺏', '빼앗', '밧', '애긋', '짜드라웃', '어그솟', '들솟', '씻', '빨가벗', '깃', '벌거벗', '엇', '되빼앗', '웃', '앗', '헐벗', '용솟', '덧솟', '발가벗', '뻘거벗', '날솟', '치솟']
    reg_d = ['맞받', '내딛', '내리받', '벋', '뒤닫', '주고받', '공얻', '무뜯', '물어뜯', '여닫', '그러묻', '잇닫', '덧묻', '되받', '뻗', '올리닫', '헐뜯', '들이닫', '활걷', '겉묻', '닫', '창받', '건네받', '물손받', '들이받', '강요받', '내리벋', '받', '이어받', '부르걷', '응받', '검뜯', '인정받', '내려딛', '내쏟', '내리뻗', '너름받', '세받', '내 돋', '돌려받', '쥐어뜯', '껴묻', '본받', '뒤받', '강종받', '내리닫', '떠받', '테받', '내받', '흠뜯', '두남받', '치받', '부르돋', '대받', '설굳', '처닫', '얻', '들이돋', '돋', '죄받', '쏟', '씨받', '딱장받', '치걷', '믿', '치벋', '버림받', '북돋', '딛', '치고받', '욱걷', '물려받', '뜯', '줴뜯', '넘겨받', '안받', '내뻗', '내리쏟', '벋딛', '뒤묻', '뻗딛', '치뻗', '치닫', '줄밑걷', '굳', '내닫', '내림받']
    reg_h = ['들이좋', '터놓', '접어놓', '좋', '풀어놓', '내쌓', '꼴좋', '치쌓', '물어넣', '잇닿', '끝닿', '그러넣', '뽕놓', '낳', '내리찧', '힘닿', '내려놓', '세놓', '둘러놓', '들놓', '맞찧', '잡아넣', '돌라쌓', '덧쌓', '갈라땋', '주놓', '갈라놓', '들이닿', '집어넣', '닿', '의좋', '막놓', '내놓', '들여놓', '사놓', '썰레놓', '짓찧', '벋놓', '찧', '침놓', '들이찧', '둘러쌓', '털어놓', '담쌓', '돌라놓', '되잡아넣', '끌어넣', '덧놓', '맞닿', '처넣', '빻', '뻥놓', '내리쌓', '곱놓', '설레발놓', '우겨넣', '놓', '수놓', '써넣', '널어놓', '덮쌓', '연닿', '헛놓', '돌려놓', '되쌓', '욱여넣', '앗아넣', '올려놓', '헛방놓', '날아놓', '뒤놓', '업수놓', '가로놓', '맞놓', '펴놓', '내켜놓', '쌓', '끙짜놓', '들이쌓', '겹쌓', '기추놓', '넣', '불어넣', '늘어놓', '긁어놓', '어긋놓', '앞넣', '눌러놓', '땋', '들여쌓', '빗놓', '사이좋', '되놓', '헛불놓', '몰아넣', '먹놓', '밀쳐놓', '살닿', '피새놓', '빼놓', '하차놓', '틀어넣']
    reg_r = ['우러르', '따르', '붙따르', '늦치르', '다다르', '잇따르', '치르']

    # 활용 유형 상수
    CT_REG   = 0  # 규칙동사
    CT_D_IRR = 1  # ㄷ 불규칙
    CT_R_IRR = 2  # 르 불규칙
    CT_B_IRR = 3  # ㅂ 불규칙
    CT_S_IRR = 4  # ㅅ 불규칙
    CT_H_IRR = 5  # ㅎ 불규칙

    # 문체 상수
    M_DECLARATIVE   = 0  # 평서문
    M_INQUISITIVE   = 1  # 의문문
    M_IMPERATIVE    = 2  # 명령문
    M_PROPOSITIVE   = 3  # 청유문
    M_NOMINAL       = 4  # 명사형
    M_ADJECTIVAL    = 5  # 관형사형

    # 시제 상수
    T_PRESENT       = 0  # 현재
    T_PAST          = 1  # 과거
    T_FUTURE        = 2  # 미래 (ㄹ것)
    T_CONDITIONAL   = 3  # 조건/미래 (겠)
    T_PLUPERFECT    = 4  # 대과거

    # 격식/높임 상수
    F_INFORMAL      = 0
    F_FORMAL        = 1

    H_LOW           = 0
    H_HIGH          = 1

    # 축약 규칙 (정규표현식)
    rules = [
        (r'ㅏ(\.?)ㅏ', r'ㅏ\1'),
        (r'ㅓ(\.?)ㅓ', r'ㅓ\1'),
        (r'ㅓ(\.?)ㅣ', r'ㅐ\1'),
        (r'ㅏ(\.?)ㅣ', r'ㅐ\1'),
        (r'ㅑ(\.?)ㅣ', r'ㅒ\1'),
        (r'ㅣ(\.?)ㅓ', r'ㅕ\1'),
        (r'야요', r'에요'),
        (r'ㄹ\.(는|십|세|니|ㄴ|ㄹ|ㅂ)', r'\1.'),
        (r'ㄹ\.ㅁ', r'ㄻ'),
        (r'ㅿ', r''),
        (r'ㅭ', r'ㄹ')
    ]

    adv_rules = [
        (r'ㅐ(\.?)ㅓ', r'ㅐ\1'),
        (r'ㅡ(\.?)ㅓ', r'ㅓ\1'),
        (r'ㅜ(\.?)ㅓ', r'ㅝ\1'),
        (r'ㅗ(\.?)ㅏ', r'ㅘ\1'),
        (r'ㅚ(\.?)ㅓ', r'ㅙ\1'),
        (r'ㅙ(\.?)ㅓ', r'ㅙ\1'),
        (r'ㅘ(\.?)ㅓ', r'ㅘ\1'),
        (r'ㅝ(\.?)ㅓ', r'ㅝ\1'),
        (r'ㅡ(\.?)ㅏ', r'ㅏ\1'),
        (r'ㅒ(\.?)ㅓ', r'ㅒ\1'),
        (r'ㅔ(\.?)ㅓ', r'ㅔ\1'),
        (r'ㅕ(\.?)ㅓ', r'ㅕ\1'),
        (r'ㅏ(\.?)ㅕ', r'ㅐ\1'),
        (r'ㅖ(\.?)ㅓ', r'ㅖ\1'),
        (r'ㅞ(\.?)ㅓ', r'ㅞ\1')
    ]

    def __init__(self, base, contraction=True):
        # 입력된 동사 기본형에서 종결 어미 "다"를 제거 (UTF-8 기준 한 글자)
        if base.endswith('다'):
            base = base[:-1]
        self.base = base
        self.b = self.hgDecomposeLast(base)
        self.ct = self.getConjType(base, self.b)
        self.pn = self.getPNType()
        self.contraction = contraction
        self.cache = {}

    # UTF-8 관련 함수는 Python 내장 함수 사용
    def utf8_strlen(self, s):
        return len(s)

    def utf8_substr(self, s, p, l):
        return s[p:p+l]

    def utf8_charAt(self, s, num):
        return s[num]

    def utf8_ord(self, ch):
        return ord(ch)

    def utf8_chr(self, u):
        return chr(u)

    @staticmethod
    def endsWith(haystack, needle):
        return haystack.endswith(needle)

    @staticmethod
    def ends_in_array(array, needle):
        for v in array:
            if KConjugator.endsWith(needle, v):
                return v
        return ''

    def hgDecomposeLast(self, s):
        if self.utf8_strlen(s) == 0:
            return s
        return s[:-1] + self.hgDecompose(s[-1])

    def hgDecompose(self, s):
        result = ""
        for ch in s:
            code = self.utf8_ord(ch) - 0xAC00
            if 0 <= code < 11172:
                cho_idx = code // 588
                jung_idx = (code % 588) // 28
                jong_idx = code % 28
                result += self.u_cho[cho_idx] + self.u_jung[jung_idx] + self.u_jong[jong_idx]
            else:
                result += ch
        return result

    def hgCompose(self, cho, jung, jong):
        try:
            i0 = self.u_cho.index(cho)
            i1 = self.u_jung.index(jung)
            i2 = self.u_jong.index(jong)
        except ValueError:
            return None
        return self.utf8_chr(i0 * 588 + i1 * 28 + i2 + 0xAC00)

    def hgEndsWithVowel(self, s):
        if not s:
            return False
        e = s[-1]
        if e in self.u_jung:
            return True
        code = self.utf8_ord(e) - 0xAC00
        if 0 <= code < 11172:
            jong_idx = code % 28
            return jong_idx == 0
        return False

    def getConjType(self, base, b):
        if b.endswith('ㄷ') and not self.ends_in_array(self.reg_d, base):
            return self.CT_D_IRR
        if b.endswith('ㄹㅡ') and not self.ends_in_array(self.reg_r, base):
            return self.CT_R_IRR
        if b.endswith('ㅂ') and not self.ends_in_array(self.reg_b, base):
            return self.CT_B_IRR
        if b.endswith('ㅅ') and not self.ends_in_array(self.reg_s, base):
            return self.CT_S_IRR
        if b.endswith('ㅎ') and not self.ends_in_array(self.reg_h, base):
            return self.CT_H_IRR
        return self.CT_REG

    def setContraction(self, cont=True):
        self.contraction = cont

    def getPNType(self):
        for i in range(self.utf8_strlen(self.base)-1, -1, -1):
            ch = self.utf8_charAt(self.base, i)
            code = self.utf8_ord(ch) - 0xAC00
            if 0 <= code < 11172:
                jung_idx = (code % 588) // 28
                jong_idx = code % 28
                if self.ct == self.CT_B_IRR and ch not in ['곱', '돕']:
                    return 0
                if ch in ['뜨', '쓰', '트']:
                    return 0
                if jung_idx in [0, 2, 8]:
                    return 1
                if not (jung_idx == 18 and jong_idx == 0):
                    return 0
        return 0

    def makeBase(self):
        return self.b

    def makeBase2(self):
        if 'base2' in self.cache:
            return self.cache['base2']
        if self.base == '아니':
            r = 'ㅇㅏㄴㅣ'
        elif self.base == '뵙':
            r = 'ㅂㅚ'
        elif self.base == '푸':
            r = 'ㅍㅓ'
        elif self.ct == self.CT_H_IRR:
            r = self.b[:-1] + 'ㅣ'
        elif self.ct == self.CT_B_IRR:
            if self.base in ['돕', '곱']:
                r = self.b[:-1] + 'ㅗ'
            else:
                r = self.b[:-1] + 'ㅜ'
        elif self.ct == self.CT_D_IRR:
            r = self.b[:-1] + 'ㅭ'
        elif self.ct == self.CT_S_IRR:
            r = self.b[:-1] + 'ㅿ'
        else:
            r = self.b
        self.cache['base2'] = r
        return r

    def makeBase3(self):
        if 'base3' in self.cache:
            return self.cache['base3']
        if self.base == '아니':
            r = 'ㅇㅏㄴㅣ'
        elif self.base == '뵙':
            r = 'ㅂㅚ'
        elif self.base == '푸':
            r = 'ㅍㅜ'
        elif self.ct == self.CT_H_IRR:
            r = self.b[:-1]
        elif self.ct == self.CT_B_IRR:
            r = self.b[:-1] + 'ㅜ'
        else:
            r = self.makeBase2()
        self.cache['base3'] = r
        return r

    def makeTBase(self, tense):
        key = 'B' + str(tense)
        if key in self.cache:
            return self.cache[key]
        if tense == self.T_PRESENT:
            r = self.makeBase()
        elif tense == self.T_PAST:
            if self.base == '이' or self.base.endswith('아니'):
                r = self.b + '었'
            else:
                r = self.makeD(0, 0, 0) + 'ㅆ'
        elif tense == self.T_FUTURE:
            j = self.hgEndsWithVowel(self.makeBase3()) or self.makeBase3().endswith('ㄹ')
            r = self.makeBase3() + '.' + ('ㄹ' if j else '을') + ' 것ㅣ'
        elif tense == self.T_CONDITIONAL:
            r = self.makeBase() + '겠'
        elif tense == self.T_PLUPERFECT:
            r = self.makeTBase(self.T_PAST) + '었'
        self.cache[key] = r
        return r

    def makeSV(self):
        pn = 'ㅏ' if self.pn else 'ㅓ'
        if self.base.endswith('하'):
            r = self.b + 'ㅕ'
        elif self.ct == self.CT_R_IRR:
            if self.base.endswith('푸르') or self.base.endswith('이르'):
                r = self.b + 'ㄹ' + pn
            else:
                r = self.b[:-1] + 'ㄹ' + pn
        elif self.ct == self.CT_H_IRR:
            r = self.makeBase2()
        else:
            r = self.makeBase2() + '.' + pn
        return r

    def makeD(self, t, f, h):
        key = 'D' + str(t) + str(f) + str(h)
        if key in self.cache:
            return self.cache[key]
        if t == 0 and f == 0 and h == 0:
            if self.base == '이' or self.base.endswith('아니'):
                r = self.b + '야'
            else:
                r = self.makeSV()
        elif f == 0 and h == 1:
            r = self.makeD(t, 0, 0) + '요'
        else:
            r = self.makeTBase(t)
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            if f == 0 and h == 0:
                r += '.' + ('어' if t != self.T_FUTURE else '야')
            elif h == 0:
                r += '.' + ('ㄴ다' if (t == self.T_PRESENT and j) else ('는다' if t == self.T_PRESENT else '다'))
            else:
                r += '.' + ('ㅂ니다' if j else '습니다')
        self.cache[key] = r
        return r

    def makeQ(self, t, f, h):
        key = 'Q' + str(t) + str(f) + str(h)
        if key in self.cache:
            return self.cache[key]
        if f == 0:
            r = self.makeD(t, f, h)
        else:
            r = self.makeTBase(t)
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            if h == 0:
                r += '.니'
            else:
                r += '.' + ('ㅂ니까' if j else '습니까')
        self.cache[key] = r
        return r

    def makeI(self, t, f, h):
        key = 'I' + str(t) + str(f) + str(h)
        if key in self.cache:
            return self.cache[key]
        if f == 0 and h == 0:
            r = self.makeD(0, 0, 0)
        elif f == 0 and h == 1:
            r = self.makeBase3()
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('' if j else '으') + '세요'
        elif f == 1 and h == 0:
            if self.base == '이' or self.base.endswith('아니'):
                r = self.b + '어라'
            else:
                r = self.makeD(0, 0, 0) + '라'
        elif f == 1 and h == 1:
            r = self.makeBase3()
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('' if j else '으') + '십시오'
        self.cache[key] = r
        return r

    def makeP(self, t, f, h):
        key = 'P' + str(t) + str(f) + str(h)
        if key in self.cache:
            return self.cache[key]
        if f == 0:
            r = self.makeD(0, 0, h)
        elif h == 0:
            r = self.makeBase() + '자'
        elif h == 1:
            r = self.makeBase3()
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('ㅂ시다' if j else '읍시다')
        self.cache[key] = r
        return r

    def makeN(self, t, f):
        key = 'N' + str(t) + str(f)
        if key in self.cache:
            return self.cache[key]
        if f == 0:
            if t == 0:
                r = self.makeBase3()
            elif t == self.T_PAST or t == self.T_CONDITIONAL:
                r = self.makeTBase(t)
            else:
                r = ''
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('ㅁ' if j else '음')
        else:
            if t != self.T_FUTURE:
                r = self.makeTBase(t) + '기'
            else:
                r = ''
        self.cache[key] = r
        return r

    def makeA(self, t):
        key = 'A' + str(t)
        if key in self.cache:
            return self.cache[key]
        if t == 0:
            r = self.makeBase() + '.는'
        elif t == 1:
            r = self.makeBase3()
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('ㄴ' if j else '은')
        elif t == 2:
            r = self.makeBase3()
            j = self.hgEndsWithVowel(r) or r.endswith('ㄹ')
            r += '.' + ('ㄹ' if j else '을') + '래'
        else:
            r = ''
        self.cache[key] = r
        return r

    def applyPR(self, d):
        for pattern, repl in self.rules:
            d = re.sub(pattern, repl, d)
        if self.contraction:
            for pattern, repl in self.adv_rules:
                d = re.sub(pattern, repl, d)
        return d

    def composite(self, s):
        result = ""
        cho = ""
        jung = ""
        jong = ""
        for c in s:
            if not cho:
                if c in self.u_jong or c in self.u_cho:
                    cho = c
                elif c in self.u_jung:
                    cho = 'ㅇ'
                    jung = c
                elif c != '.':
                    result += c
            elif cho and not jung:
                if c in self.u_jong or c in self.u_cho:
                    result += cho
                    cho = c
                elif c in self.u_jung:
                    jung = c
                else:
                    result += cho
                    if c != '.':
                        result += c
                    cho = ""
            elif cho and jung and not jong:
                if c in self.u_jong:
                    jong = c
                elif c in self.u_cho:
                    comp = self.hgCompose(cho, jung, '')
                    result += comp if comp else (cho + jung)
                    cho = c
                    jung = ""
                elif c in self.u_jung:
                    comp = self.hgCompose(cho, jung, '')
                    result += comp if comp else (cho + jung)
                    cho = 'ㅇ'
                    jung = c
                elif c == '.':
                    pass
                else:
                    comp = self.hgCompose(cho, jung, '')
                    result += (comp if comp else (cho + jung)) + c
                    cho = ""
                    jung = ""
            else:
                if c in self.u_jong or c in self.u_cho:
                    comp = self.hgCompose(cho, jung, jong)
                    result += comp if comp else (cho + jung + jong)
                    cho = c
                    jung = ""
                    jong = ""
                elif c in self.u_jung:
                    comp = self.hgCompose(cho, jung, jong)
                    result += comp if comp else (cho + jung + jong)
                    cho = jong
                    jung = c
                    jong = ""
                else:
                    comp = self.hgCompose(cho, jung, jong)
                    result += comp if comp else (cho + jung + jong)
                    if c != '.':
                        result += c
                    cho = ""
                    jung = ""
                    jong = ""
        if cho and jung:
            comp = self.hgCompose(cho, jung, jong)
            result += comp if comp else (cho + jung + jong)
        elif cho:
            result += cho
        elif jung:
            comp = self.hgCompose('ㅇ', jung, jong)
            result += comp if comp else ('ㅇ' + jung + jong)
        return result

    def conjugate(self, m, t, f=0, h=0):
        if m == self.M_DECLARATIVE:
            return self.composite(self.applyPR(self.makeD(t, f, h)))
        if m == self.M_INQUISITIVE:
            return self.composite(self.applyPR(self.makeQ(t, f, h)))
        if m == self.M_IMPERATIVE:
            return self.composite(self.applyPR(self.makeI(t, f, h)))
        if m == self.M_PROPOSITIVE:
            return self.composite(self.applyPR(self.makeP(t, f, h)))
        if m == self.M_NOMINAL:
            return self.composite(self.applyPR(self.makeN(t, f)))
        if m == self.M_ADJECTIVAL:
            return self.composite(self.applyPR(self.makeA(t)))
        return ""

# --- 사용 예시 ---
if __name__ == '__main__':
    # 예를 들어, "틀다"에서 종결 어미 "다"를 제거한 "틀"을 입력
    verb = "켜주"
    kc = KConjugator(verb)

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

    # 모든 가능한 조합을 출력
    for mood in range(6):  # 0 ~ 5
        for tense in range(5):  # 0 ~ 4
            for formality in range(2):  # 0, 1
                # 명사형과 관형사형은 honorific이 영향을 주지 않으므로 honorific=0만 사용
                if mood in [KConjugator.M_NOMINAL, KConjugator.M_ADJECTIVAL]:
                    result = kc.conjugate(mood, tense, formality, 0)
                    print(f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]} => {result}")
                else:
                    for honorific in range(2):  # 0, 1
                        result = kc.conjugate(mood, tense, formality, honorific)
                        print(f"{moodNames[mood]}, {tenseNames[tense]}, {formalNames[formality]}, {honorificNames[honorific]} => {result}")
