languages = {
    "Ancient Greek": {
        "code": "grc",
        "example": "εἴησαν δ’ ἄν οὗτοι Κρῆτες",
        "source": "the [Diorisis corpus](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256), totaling 9,058,227 tokens",
        "ud": "[UD_Ancient_Greek-PROEIL](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL), v2.9, totaling 213,999 tokens,",
    },
    "Coptic": {
        "code": "cop",
        "example": "ⲁⲗⲗⲁ ⲁⲛⲟⲕ ⲁⲓⲥⲉⲧⲡⲧⲏⲩⲧⲛ ·",
        "source": "version 4.2.0 of the [Coptic SCRIPTORIUM corpus](https://github.com/copticscriptorium/corpora), totaling 970,642 tokens",
        "ud": "[UD_Coptic_Scriptorium](https://github.com/UniversalDependencies/UD_Coptic-Scriptorium), v2.9, totaling 48,632 tokens,",
    },
    "Indonesian": {
        "code": "ind",
        "example": "Lalu bagaimana dengan kisah cinta Mutia dan Fadel?",
        "source": "a June 2022 dump of Indonesian Wikipedia, downsampled to 1,439,772 tokens",
        "ud": "[UD_Indonesian-GSD](https://github.com/UniversalDependencies/UD_Indonesian-GSD), v2.10, totaling 122,021 tokens,"
    },
    "Maltese": {
        "code": "mlt",
        "example": "Hu rrifjuta l-akkużi li l-Gvern Malti qed jimxi bi djufija.",
        "source": "a February 2022 dump of Maltese Wikipedia, totaling 2,113,223 tokens",
        "ud": "[UD_Maltese-GSD](https://github.com/UniversalDependencies/UD_Maltese-MUDT), v2.9, totaling 44,162 tokens,"
    },
    "Uyghur": {
        "code": "uig",
        "example": "ھﺎﻳﺎﺗ ﺕﻮﻏﺭﻰﻗﻯڭﻥﻯڭ ﺉۆﻡۈﺭ ﻱﻰﻠﺗﻯﺯﻰﻧﻯ ﻕۇﺮﺘﺗەﻙ ﺉﺍۋﺎﻳﻼﭘ ﻱەﻲﻣەﻥ.",
        "source": "a February 2022 dump of Uyghur Wikipedia, totaling 2,401,445 tokens",
        "ud": "[UD_Uyghur-UDT](https://github.com/UniversalDependencies/UD_Uyghur-UDT), v2.9, totaling 40,236 tokens,"
    },
    "Tamil": {
        "code": "tam",
        "example": "தொழிலாளர்களுடன் பேச்சுவார்த்தை நடத்தி, சுமூக தீர்வு காணப்படவில்லை.",
        "source": "a June 2022 dump of Tamil Wikipedia, downsampled to 1,429,735 tokens",
        "ud": "[UD_Tamil-TTB](https://github.com/UniversalDependencies/UD_Tamil-TTB), v2.9, totaling 9,581 tokens,"
    },
    "Wolof": {
        "code": "wol",
        "example": "Looloo taxoon ñuy réew yu naat.",
        "source": "a February 2022 dump of Uyghur Wikipedia, totaling 517,237 tokens",
        "ud": "[UD_Wolof-WDT](https://github.com/UniversalDependencies/UD_Wolof-WDT), v2.9, totaling 44,258 tokens,"
    }
}

hf_template = """
---
language: {{code}}
widget:
- text: {{example}}
---

This is a [MicroBERT](https://github.com/lgessler/microbert) model for {{language}}.

* Its suffix is **-{{suffix}}**, which means that it was pretrained using supervision from {{suffix_expl}}.
* The unlabeled {{language}} data was taken from {{source}}. 
* The UD treebank {{ud}} was used for labeled data.

Please see [the repository](https://github.com/lgessler/microbert) and 
[the paper](https://github.com/lgessler/microbert/raw/master/MicroBERT__MRL_2022_.pdf) for more details.

"""


for k, v in languages.items():
    for suffix, suffix_expl in [("m", "masked language modeling"),
                                ("mx", "masked language modeling and XPOS tagging"),
                                ("mxp", "masked language modeling, XPOS tagging, and UD dependency parsing")]:
        print()
        print(
            hf_template
            .replace("{{language}}", k)
            .replace("{{suffix}}", suffix)
            .replace("{{suffix_expl}}", suffix_expl)
            .replace("{{source}}", v["source"])
            .replace("{{ud}}", v["ud"])
            .replace("{{code}}", v["code"])
            .replace("{{example}}", v["example"])
        )
        input()
