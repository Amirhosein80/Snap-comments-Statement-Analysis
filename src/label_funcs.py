from snorkel.labeling import labeling_function

ABSTAIN = -1
POSITIVE = 1
NEGATIVE = 0

POSITIVE_KEYWORDS = [
    "عالی",
    "خوشمزه",
    "خوب",
    "به موقع",
    "لذیذ",
    "تمیز",
    "خوبه",
    "سریع",
    "زود",
    "راضی",
    "محشر",
    "بی‌_نظیر",
    "داغ",
    "گرم",
    "خوش",
    "دل‌_چسب",
    "کارآمد",
    "مناسب",
    "پیشنهاد_می‌کنم",
    "همیشه",
    "زیاد",
    "چسبید",
    "فوق_العاده",
    "سالم",
    "عالیه",
]

NEGATIVE_KEYWORDS = [
    "بد",
    "بی‌_کیفیت",
    "ناامید",
    "اشتباه",
    "تاخیر",
    "کثیف",
    "گرون",
    "بی‌_مزه",
    "افتضاح",
    "متاسفانه",
    "کم",
    "پایین",
    "بدترین",
    "ریخته",
    "ناراضی",
    "پشیمان",
    "بی‌ارزش",
    "بدقولی",
    "نبود",
    "هیچ",
    "بجاش",
    "بجای",
    "شاید",
    "بی‌_توجهی" "منتظر",
    "دور",
    "نداشت",
    "کمتر",
    "ندارین",
]


@labeling_function()
def lf_positive_keywords(comment):
    return (
        POSITIVE
        if any(
            word in comment.processed_comment.split(" ") for word in POSITIVE_KEYWORDS
        )
        else ABSTAIN
    )


@labeling_function()
def lf_negative_keywords(comment):
    return (
        NEGATIVE
        if any(
            word in comment.processed_comment.split(" ") for word in NEGATIVE_KEYWORDS
        )
        else ABSTAIN
    )


@labeling_function()
def lf_question_mark(comment):
    if "؟" in comment.processed_comment:
        return NEGATIVE
    else:
        return ABSTAIN


LABEL_FUNCS = [lf_negative_keywords, lf_positive_keywords, lf_question_mark]
