"""Insert the Borda ranking chart as Hinh 6.9 after Hinh 6.8 and bold every figure caption."""
import copy
import sys

import docx
from docx.shared import Inches

src, chart, dst = sys.argv[1:4]
CAPTION = "Hình 6.9. Thuật toán có Borda count cao nhất theo độ thưa dữ liệu (AIU) và nhóm người dùng"

d = docx.Document(src)
paras = d.paragraphs

cap68 = next(p for p in paras if p.text.strip().startswith("Hình 6.8."))
img68 = paras[[p._p for p in paras].index(cap68._p) - 1]

# image paragraph: same paragraph formatting as the Hinh 6.8 image, new picture
img_p = copy.deepcopy(img68._p)
for r in img_p.findall(docx.oxml.ns.qn("w:r")):
    img_p.remove(r)
cap68._p.addnext(img_p)
img_para = docx.text.paragraph.Paragraph(img_p, cap68._parent)
img_para.add_run().add_picture(chart, width=Inches(6.5))

# caption paragraph: clone of the Hinh 6.8 caption with new text
cap_p = copy.deepcopy(cap68._p)
img_p.addnext(cap_p)
cap_para = docx.text.paragraph.Paragraph(cap_p, cap68._parent)
for r in cap_para.runs[1:]:
    r._r.getparent().remove(r._r)
cap_para.runs[0].text = CAPTION

n = 0
for p in d.paragraphs:
    if p.text.strip().startswith("Hình "):
        for r in p.runs:
            r.bold = True
        n += 1

d.save(dst)
print("bolded captions:", n)
