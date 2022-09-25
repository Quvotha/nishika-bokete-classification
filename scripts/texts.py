import re
import unicodedata


def cleanse(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub("[ ]+", " ", text)
    text = re.sub("[ー]+", "ー", text)
    text = re.sub("[!]+", "!", text)
    text = re.sub("[w]{3,}", "ww", text)
    return text


if __name__ == "__main__":
    testdata = (
        ("ルーレットスタート！          シカカカカカカカカカ…", "ルーレットスタート! シカカカカカカカカカ..."),
        (" ソソミ", "ソソミ"),
        ("こんにちわ　こんにちわ　　こんにちわ  こんにちわ", "こんにちわ こんにちわ こんにちわ こんにちわ"),
        ("めーーっちゃ、気ぃ使こたーーー", "めーっちゃ、気ぃ使こたー"),
        ("アンパンマン！新しい顔よっ！！", "アンパンマン!新しい顔よっ!"),
        ("ハハハハ、滑らねーwww", "ハハハハ、滑らねーww"),
        ("ハハハハ、滑らねーwwwwww", "ハハハハ、滑らねーww"),
    )
    for testdata_ in testdata:
        input_, expected = testdata_
        output = cleanse(input_)
        assert output == expected, (input_, expected, output)
