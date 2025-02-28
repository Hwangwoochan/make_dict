import pandas as pd
from src.worker import convert

def main(word: str = None) -> None:
    if word is None:
        while True:
            usr_input = input("Enter the 한글 to convert (q to quit): ").lower()
            if usr_input.lower() == 'q':
                break
            res = convert(usr_input, sep=' ')
            print(res)
            return res, usr_input
    else:
        return convert(word, sep=' ')


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


if __name__ == '__main__':

    col = input("1. 주어, 2. 동사, 3. 문장 : ")

    match col:
        case "1":
            clean = pd.read_csv("./subject.csv", header=None)

            # 첫 행 전부 숫자로 바꾸되, 변환 불가능한건 NaN -> 0
            row0 = pd.to_numeric(clean.iloc[0], errors='coerce').fillna(0).astype(int)
            clean.iloc[0] = row0

            sub_lengths = clean.iloc[0].dropna().values.astype(int)
            sub = clean.iloc[1:].reset_index(drop=True)

            text, han = main()
            length = text.count(" ") + 1

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

            # 저장 전, 첫 행 다시 동일 처리(혹시 None이 들어갔을 수도 있으므로)
            row0 = pd.to_numeric(clean.iloc[0], errors='coerce').fillna(0).astype(int)
            clean.iloc[0] = row0

            clean.to_csv("./subject.csv", encoding="UTF-8-SIG", index=False, header=False)
            print(f"Updated at row={col_index+1}, col={row_index} (한글 col={row_index+1})")


        case "2":
            clean = pd.read_csv("./intent.csv", header=None)

            row0 = pd.to_numeric(clean.iloc[0], errors='coerce').fillna(0).astype(int)
            clean.iloc[0] = row0

            sub_lengths = clean.iloc[0].dropna().values.astype(int)
            sub = clean.iloc[1:].reset_index(drop=True)

            text, han = main()
            length = text.count(" ") + 1

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

            clean.to_csv("./intent.csv", encoding="UTF-8-SIG", index=False, header=False)
            print(f"Updated at row={col_index+1}, col={row_index} (한글 col={row_index+1})")
        case "3":
            text, han = main()  # 예: text = "pangbulggeuljo", han = "방불꺼줘"
            
            import csv
            filename = "sentence.csv"
            with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # 한 행에 두 문자열을 넣어서 저장합니다.
                writer.writerow([text, han])
            print(f"{filename} 파일이 성공적으로 저장되었습니다.")
        case _:
            print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
